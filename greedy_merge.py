"""
greedy_merge.py
===============
Builds connected components from LSH candidate pairs using union-find,
picks one representative per cluster, and writes a final dedup manifest.

Reads from:  dedup_results.db (hash_duplicates + lsh_candidates)
Writes to:   dedup_results.db (dedup_manifest table)

The manifest has one row per original document with columns:
  - base_path:      original document path
  - cluster_id:     integer cluster ID
  - is_representative: 1 if this is the keeper, 0 if duplicate
  - duplicate_of:   path of the representative (NULL if is_representative)
  - source:         how the duplicate was detected ('hash_exact', 'hash_normalized', 'lsh')

Usage:
  python greedy_merge.py
  python greedy_merge.py --min-jaccard 0.8   # only merge pairs above this threshold
"""

import argparse
import sqlite3
import time


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups


def main():
    parser = argparse.ArgumentParser(description="Greedy merge via union-find on LSH pairs")
    parser.add_argument(
        "--min-jaccard", type=float, default=0.0,
        help="Only merge LSH pairs with jaccard_est >= this value (default: 0.0, use all)"
    )
    args = parser.parse_args()

    RESULTS_DB = "data/dedup_results.db"
    SOURCE_DB  = "data/Baytown_chunks.db"

    conn = sqlite3.connect(RESULTS_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Build union-find from hash groups
    # ------------------------------------------------------------------
    print("  Loading hash duplicate groups...")
    uf = UnionFind()

    hash_rows = c.execute(
        "SELECT group_id, base_path FROM hash_duplicates ORDER BY group_id"
    ).fetchall()

    hash_groups: dict[int, list[str]] = {}
    for gid, path in hash_rows:
        hash_groups.setdefault(gid, []).append(path)

    # Track which source detected each document's duplication
    doc_source: dict[str, str] = {}

    for gid, paths in hash_groups.items():
        for p in paths:
            doc_source[p] = "hash"
        for i in range(1, len(paths)):
            uf.union(paths[0], paths[i])

    hash_doc_count = len(doc_source)
    print(f"  Hash groups: {len(hash_groups):,} groups, {hash_doc_count:,} documents")
    del hash_groups

    # ------------------------------------------------------------------
    # Step 2: Merge LSH candidate pairs into union-find
    # ------------------------------------------------------------------
    print(f"  Loading LSH candidate pairs (min_jaccard={args.min_jaccard})...")

    if args.min_jaccard > 0:
        lsh_rows = c.execute(
            "SELECT path_a, path_b FROM lsh_candidates WHERE jaccard_est >= ?",
            (args.min_jaccard,)
        ).fetchall()
    else:
        lsh_rows = c.execute(
            "SELECT path_a, path_b FROM lsh_candidates"
        ).fetchall()

    print(f"  LSH pairs to merge: {len(lsh_rows):,}")

    for path_a, path_b in lsh_rows:
        uf.union(path_a, path_b)
        if path_a not in doc_source:
            doc_source[path_a] = "lsh"
        if path_b not in doc_source:
            doc_source[path_b] = "lsh"

    del lsh_rows

    # ------------------------------------------------------------------
    # Step 3: Also include singleton documents (no duplicates at all)
    # ------------------------------------------------------------------
    print("  Finding singleton documents...")
    source_conn = sqlite3.connect(SOURCE_DB)
    sc = source_conn.cursor()
    sc.execute("SELECT DISTINCT filename FROM chunks")

    all_bases = set()
    for (fn,) in sc.fetchall():
        base, sep, idx = fn.rpartition("_")
        all_bases.add(base if (sep and idx.isdigit()) else fn)

    source_conn.close()

    singletons = all_bases - set(uf.parent.keys())
    for s in singletons:
        uf.find(s)  # register as its own component
    print(f"  Singletons (no duplicates): {len(singletons):,}")

    # ------------------------------------------------------------------
    # Step 4: Extract components, pick representatives
    # ------------------------------------------------------------------
    components = uf.components()
    print(f"  Total clusters: {len(components):,}")

    # Pick representative: shortest path name, then alphabetically first
    # (heuristic: shorter names tend to be the "original")
    def pick_rep(paths: list[str]) -> str:
        return min(paths, key=lambda p: (len(p), p))

    # ------------------------------------------------------------------
    # Step 5: Write manifest
    # ------------------------------------------------------------------
    print("  Writing dedup_manifest table...")

    c.execute("DROP TABLE IF EXISTS dedup_manifest")
    c.execute("""
        CREATE TABLE dedup_manifest (
            base_path         TEXT PRIMARY KEY,
            cluster_id        INTEGER NOT NULL,
            is_representative INTEGER NOT NULL,
            duplicate_of      TEXT,
            source            TEXT
        )
    """)

    batch = []
    total_reps = 0
    total_dups = 0

    for cluster_id, (root, paths) in enumerate(sorted(components.items())):
        rep = pick_rep(paths)
        for p in paths:
            if p == rep:
                batch.append((p, cluster_id, 1, None, doc_source.get(p)))
                total_reps += 1
            else:
                batch.append((p, cluster_id, 0, rep, doc_source.get(p)))
                total_dups += 1

            if len(batch) >= 10_000:
                c.executemany(
                    "INSERT INTO dedup_manifest VALUES (?,?,?,?,?)", batch
                )
                conn.commit()
                batch.clear()

    if batch:
        c.executemany("INSERT INTO dedup_manifest VALUES (?,?,?,?,?)", batch)
        conn.commit()

    c.execute("CREATE INDEX IF NOT EXISTS idx_dm_cluster ON dedup_manifest(cluster_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_dm_rep ON dedup_manifest(is_representative)")
    conn.commit()

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  {'='*50}")
    print(f"  Dedup Summary")
    print(f"  {'='*50}")
    print(f"  Total documents:    {total_reps + total_dups:,}")
    print(f"  Unique (keep):      {total_reps:,}")
    print(f"  Duplicates (drop):  {total_dups:,}")
    print(f"  Reduction:          {total_dups / (total_reps + total_dups) * 100:.1f}%")
    print(f"  Elapsed:            {elapsed:.1f}s")

    # Cluster size distribution
    sizes = [len(paths) for paths in components.values()]
    size_1 = sum(1 for s in sizes if s == 1)
    size_2_5 = sum(1 for s in sizes if 2 <= s <= 5)
    size_6_20 = sum(1 for s in sizes if 6 <= s <= 20)
    size_21_100 = sum(1 for s in sizes if 21 <= s <= 100)
    size_100p = sum(1 for s in sizes if s > 100)

    print(f"\n  Cluster size distribution:")
    print(f"    Size 1 (unique):  {size_1:,}")
    print(f"    Size 2-5:         {size_2_5:,}")
    print(f"    Size 6-20:        {size_6_20:,}")
    print(f"    Size 21-100:      {size_21_100:,}")
    print(f"    Size 100+:        {size_100p:,}")

    if size_100p > 0:
        print(f"\n  Largest clusters:")
        big = sorted(
            [(len(paths), pick_rep(paths)) for paths in components.values()],
            reverse=True
        )[:10]
        for sz, rep in big:
            print(f"    {sz:>6,} docs  rep: {rep}")

    conn.close()
    print(f"\n  Results written -> {RESULTS_DB} (dedup_manifest table)")
    print(f"  Query keepers:  SELECT base_path FROM dedup_manifest WHERE is_representative = 1")


if __name__ == "__main__":
    main()