"""
dedup_pipeline.py
=================
Three-stage deduplication pipeline for Baytown_chunks.db
Operates at the DOCUMENT level — chunks are reassembled per source file.

Stages (run independently, state persisted between runs):
  python dedup_pipeline.py --stage hash
  python dedup_pipeline.py --stage lsh
  python dedup_pipeline.py --stage embed
  python dedup_pipeline.py --stage status

State is saved to dedup_state.json after each stage completes.
Results are written to dedup_results.db (separate from source DB).

Requirements:
  pip install datasketch sentence-transformers faiss-cpu tqdm numpy
  (use faiss-gpu instead of faiss-cpu if GPU-accelerated ANN search is preferred)
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = "data"

SOURCE_DB  = f"{DATA_DIR}/Baytown_chunks.db"
RESULTS_DB = f"{DATA_DIR}/dedup_results.db"
STATE_FILE = f"{DATA_DIR}/dedup_state.json"

# Hash stage
NORMALIZE  = True        # also compute normalized hash (whitespace collapsed)

# LSH stage
LSH_THRESHOLD  = 0.7     # Jaccard similarity threshold (lower = more candidates)
LSH_NUM_PERM   = 128     # MinHash permutations (higher = more accurate, slower)
SHINGLE_SIZE   = 5        # word n-gram size for shingling
LSH_WORKERS    = 0        # 0 = auto (cpu_count), 1 = serial (no multiprocessing)

# Embedding stage
EMBED_MODEL     = "all-MiniLM-L6-v2"
EMBED_BATCH     = 64             # documents per batch
EMBED_THRESHOLD = 0.92           # cosine similarity floor for confirmed duplicates
EMBED_TRUNCATE  = 512            # max words fed to the model per document

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"completed_stages": [], "stats": {}}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  State saved -> {STATE_FILE}")

# ---------------------------------------------------------------------------
# Results DB
# ---------------------------------------------------------------------------

def open_results_db() -> sqlite3.Connection:
    conn = sqlite3.connect(RESULTS_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS hash_duplicates (
            group_id     INTEGER NOT NULL,
            base_path    TEXT NOT NULL,
            hash_type    TEXT NOT NULL,
            content_hash TEXT NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_hd_group ON hash_duplicates(group_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_hd_path  ON hash_duplicates(base_path)")

    c.execute("""
        CREATE TABLE IF NOT EXISTS lsh_candidates (
            pair_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            path_a      TEXT NOT NULL,
            path_b      TEXT NOT NULL,
            jaccard_est REAL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_lsh_a ON lsh_candidates(path_a)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_lsh_b ON lsh_candidates(path_b)")

    c.execute("""
        CREATE TABLE IF NOT EXISTS embed_duplicates (
            pair_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            path_a     TEXT NOT NULL,
            path_b     TEXT NOT NULL,
            cosine_sim REAL NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_ed_a ON embed_duplicates(path_a)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ed_b ON embed_duplicates(path_b)")

    conn.commit()
    return conn

# ---------------------------------------------------------------------------
# Core helper: reassemble document text from chunks
# ---------------------------------------------------------------------------

def reassemble_doc(chunks_by_index: list) -> str:
    """
    chunks_by_index: [(chunk_index, text), ...]
    Returns the reconstructed document text with overlap removed.

    Uses a suffix-matching approach: hash the last N lines of the
    accumulated result, then check if the first lines of the new chunk
    match any suffix. This is O(n) instead of the naive O(n²) approach.
    """
    if not chunks_by_index:
        return ""
    if len(chunks_by_index) == 1:
        return chunks_by_index[0][1]

    chunks_by_index = sorted(chunks_by_index, key=lambda x: x[0])
    result_lines = chunks_by_index[0][1].splitlines()

    for _, chunk_text in chunks_by_index[1:]:
        new_lines = chunk_text.splitlines()
        if not new_lines:
            continue

        # Build a set of suffix hashes from the tail of result_lines.
        # For each candidate overlap length n, hash the last n lines of result
        # and compare against the first n lines of new_lines.
        max_check = min(len(result_lines), len(new_lines), 100)
        overlap_len = 0

        if max_check > 0:
            # Optimization: quick check — if the last line of result doesn't
            # appear anywhere in the first max_check lines of new_lines,
            # there's no overlap at all.
            last_line = result_lines[-1]
            candidate_positions = [
                i for i in range(max_check) if new_lines[i] == last_line
            ]

            if candidate_positions:
                # Only check overlaps that end at a matching line.
                # An overlap of length n means result[-n:] == new[:n],
                # so result[-1] == new[n-1], meaning n = position + 1.
                for pos in reversed(candidate_positions):
                    n = pos + 1
                    if n > max_check:
                        continue
                    if result_lines[-n:] == new_lines[:n]:
                        overlap_len = n
                        break

        result_lines.extend(new_lines[overlap_len:])

    return "\n".join(result_lines)


def iter_documents(source_conn: sqlite3.Connection):
    """
    Yield (base_path, full_text) for every distinct source document,
    reassembled from their chunks.

    Uses a single streaming query grouped by base_path derivation.
    """
    cursor = source_conn.cursor()
    cursor.execute("SELECT filename, text FROM chunks ORDER BY filename")

    current_base = None
    current_chunks = []

    for filename, text in cursor:
        base, sep, idx_str = filename.rpartition("_")
        if sep and idx_str.isdigit():
            base_key, chunk_idx = base, int(idx_str)
        else:
            base_key, chunk_idx = filename, 0

        if base_key != current_base:
            if current_base is not None:
                yield current_base, reassemble_doc(current_chunks)
            current_base = base_key
            current_chunks = []

        current_chunks.append((chunk_idx, text))

    if current_base is not None:
        yield current_base, reassemble_doc(current_chunks)


def count_documents(source_conn: sqlite3.Connection) -> int:
    """Count distinct base documents without loading all text."""
    c = source_conn.cursor()
    c.execute("SELECT DISTINCT filename FROM chunks")
    bases = set()
    for (fn,) in c.fetchall():
        base, sep, idx = fn.rpartition("_")
        bases.add(base if (sep and idx.isdigit()) else fn)
    return len(bases)


# ---------------------------------------------------------------------------
# STAGE 1: Hash
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def stage_hash():
    print("\n" + "="*60)
    print("STAGE 1: Hash Deduplication  (document level)")
    print("="*60)

    state = load_state()
    if "hash" in state["completed_stages"]:
        print("  Already complete. Use --reset to re-run.")
        return

    source_conn  = sqlite3.connect(SOURCE_DB)
    source_conn.execute("PRAGMA mmap_size=2147483648")  # 2 GB mmap for read perf
    results_conn = open_results_db()
    rc = results_conn.cursor()

    rc.execute("DELETE FROM hash_duplicates")
    results_conn.commit()

    total_docs = count_documents(source_conn)
    print(f"  Reassembling and hashing {total_docs:,} documents...")

    exact_map = {}
    norm_map  = {}

    t0 = time.time()

    # Use an incremental hasher so we don't double-allocate with .encode()
    # on massive document strings. For normalized, hash the normalized form
    # only — no need to hold both in memory.
    for base_path, doc_text in tqdm(
        iter_documents(source_conn), total=total_docs, desc="  Hashing",
        smoothing=0.05,   # slow down ETA jitter from document size variance
        mininterval=1.0,
    ):
        doc_bytes = doc_text.encode("utf-8", errors="replace")
        h_exact = hashlib.sha256(doc_bytes).hexdigest()
        exact_map.setdefault(h_exact, []).append(base_path)

        if NORMALIZE:
            h_norm = hashlib.sha256(
                normalize_text(doc_text).encode("utf-8", errors="replace")
            ).hexdigest()
            norm_map.setdefault(h_norm, []).append(base_path)

    elapsed = time.time() - t0
    print(f"  Hashing complete in {elapsed:.1f}s")

    group_id = 0
    exact_dup_groups = exact_dup_docs = 0

    for h, paths in exact_map.items():
        if len(paths) > 1:
            for p in paths:
                rc.execute(
                    "INSERT INTO hash_duplicates VALUES (?,?,?,?)",
                    (group_id, p, "exact", h)
                )
            group_id += 1
            exact_dup_groups += 1
            exact_dup_docs   += len(paths)

    exact_dup_path_set = {
        p
        for paths in exact_map.values() if len(paths) > 1
        for p in paths
    }
    norm_dup_groups = norm_dup_docs = 0

    if NORMALIZE:
        for h, paths in norm_map.items():
            if len(paths) > 1:
                new_paths = [p for p in paths if p not in exact_dup_path_set]
                if len(new_paths) > 1:
                    for p in new_paths:
                        rc.execute(
                            "INSERT INTO hash_duplicates VALUES (?,?,?,?)",
                            (group_id, p, "normalized", h)
                        )
                    group_id += 1
                    norm_dup_groups += 1
                    norm_dup_docs   += len(new_paths)

    results_conn.commit()

    # Free memory before next stages
    del exact_map, norm_map
    source_conn.close()
    results_conn.close()

    stats = {
        "total_documents":          total_docs,
        "exact_dup_groups":         exact_dup_groups,
        "exact_dup_documents":      exact_dup_docs,
        "normalized_dup_groups":    norm_dup_groups,
        "normalized_dup_documents": norm_dup_docs,
        "elapsed_seconds":          round(elapsed, 1),
    }

    print(f"\n  Results:")
    print(f"    Exact duplicate groups:      {exact_dup_groups:,}  ({exact_dup_docs:,} docs)")
    if NORMALIZE:
        print(f"    Normalized duplicate groups: {norm_dup_groups:,}  ({norm_dup_docs:,} docs)")

    state["completed_stages"].append("hash")
    state["stats"]["hash"] = stats
    save_state(state)
    print(f"\n  Results written -> {RESULTS_DB} (hash_duplicates table)")


# ---------------------------------------------------------------------------
# STAGE 2: LSH  (parallelized MinHash build + query)
# ---------------------------------------------------------------------------

def shingle(text: str, n: int = SHINGLE_SIZE) -> set:
    words = text.lower().split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def _compute_minhash(args):
    """
    Worker function for multiprocessing.
    Receives (base_path, doc_text, num_perm, shingle_size).
    Returns (base_path, hashvalues_as_bytes) — we serialize the numpy
    array rather than the MinHash object to avoid pickling overhead.
    """
    from datasketch import MinHash
    base_path, doc_text, num_perm, shingle_size = args
    m = MinHash(num_perm=num_perm)
    words = doc_text.lower().split()
    if len(words) < shingle_size:
        for w in words:
            m.update(w.encode("utf-8"))
    else:
        for i in range(len(words) - shingle_size + 1):
            m.update(" ".join(words[i:i+shingle_size]).encode("utf-8"))
    return base_path, m.hashvalues.tobytes()


def stage_lsh():
    print("\n" + "="*60)
    print("STAGE 2: LSH / MinHash  (document level)")
    print("="*60)

    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  ERROR: pip install datasketch")
        sys.exit(1)

    state = load_state()
    if "lsh" in state["completed_stages"]:
        print("  Already complete. Use --reset to re-run.")
        return

    n_workers = LSH_WORKERS if LSH_WORKERS > 0 else os.cpu_count()
    print(f"  Workers: {n_workers}")

    source_conn  = sqlite3.connect(SOURCE_DB)
    source_conn.execute("PRAGMA mmap_size=2147483648")
    results_conn = open_results_db()
    rc = results_conn.cursor()

    rc.execute("DELETE FROM lsh_candidates")
    results_conn.commit()

    # Build a set of representative documents: one per hash group.
    # We want to run LSH across the ~53K unique contents, not skip them.
    representatives = set()
    all_hash_paths  = set()
    try:
        rows = results_conn.execute(
            "SELECT group_id, base_path FROM hash_duplicates ORDER BY group_id, base_path"
        ).fetchall()
        groups: dict[int, list[str]] = {}
        for gid, path in rows:
            groups.setdefault(gid, []).append(path)
            all_hash_paths.add(path)
        for gid, paths in groups.items():
            representatives.add(paths[0])
        del groups
        print(f"  Hash stage found {len(representatives):,} unique content groups")
        print(f"  Using 1 representative per group for near-duplicate detection")
    except Exception:
        pass

    total_docs = count_documents(source_conn)
    target_count = len(representatives) + (total_docs - len(all_hash_paths))
    print(f"  Building MinHash index over ~{target_count:,} documents...")
    print(f"  Threshold: {LSH_THRESHOLD}  |  Permutations: {LSH_NUM_PERM}  |  Shingle: {SHINGLE_SIZE}-gram")

    t0 = time.time()

    # ---- Phase 1: Collect documents to process ----
    # Stream from DB, filter to representatives + singletons, collect text.
    # At 53K docs this is manageable in memory (we need the text for workers).
    doc_items = []
    for base_path, doc_text in tqdm(
        iter_documents(source_conn), total=total_docs, desc="  Collecting docs",
        smoothing=0.05, mininterval=1.0,
    ):
        if base_path in all_hash_paths and base_path not in representatives:
            continue
        doc_items.append((base_path, doc_text, LSH_NUM_PERM, SHINGLE_SIZE))

    source_conn.close()
    collect_time = time.time() - t0
    print(f"  Collected {len(doc_items):,} documents in {collect_time:.1f}s")

    # ---- Phase 2: Parallel MinHash computation ----
    t1 = time.time()
    minhashes = {}

    if n_workers == 1 or len(doc_items) < 500:
        # Serial path — avoid multiprocessing overhead for small jobs
        print("  Computing MinHash signatures (serial)...")
        for args in tqdm(doc_items, desc="  Hashing"):
            path, hv_bytes = _compute_minhash(args)
            m = MinHash(num_perm=LSH_NUM_PERM)
            m.hashvalues = np.frombuffer(hv_bytes, dtype=np.uint64).copy()
            minhashes[path] = m
    else:
        print(f"  Computing MinHash signatures ({n_workers} workers)...")
        # Use chunksize to reduce IPC overhead — each worker gets a batch
        chunksize = max(1, len(doc_items) // (n_workers * 4))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures_iter = pool.map(_compute_minhash, doc_items, chunksize=chunksize)
            for path, hv_bytes in tqdm(
                futures_iter, total=len(doc_items), desc="  Hashing",
                smoothing=0.05, mininterval=1.0,
            ):
                m = MinHash(num_perm=LSH_NUM_PERM)
                m.hashvalues = np.frombuffer(hv_bytes, dtype=np.uint64).copy()
                minhashes[path] = m

    del doc_items  # free the text strings
    hash_time = time.time() - t1
    print(f"  MinHash computation: {hash_time:.1f}s  |  {len(minhashes):,} signatures")

    # ---- Phase 3: LSH index insertion (serial, fast) ----
    t2 = time.time()
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM)
    for path, m in tqdm(minhashes.items(), desc="  Inserting into LSH"):
        try:
            lsh.insert(path, m)
        except ValueError:
            pass

    insert_time = time.time() - t2
    print(f"  LSH index insertion: {insert_time:.1f}s")

    # ---- Phase 4: Query + Jaccard scoring ----
    t3 = time.time()
    seen_pairs      = set()
    insert_batch    = []
    candidate_count = 0

    for path_a, m in tqdm(minhashes.items(), desc="  Querying"):
        for path_b in lsh.query(m):
            if path_b == path_a:
                continue
            pair = (min(path_a, path_b), max(path_a, path_b))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            jaccard = minhashes[path_a].jaccard(minhashes[path_b])
            insert_batch.append((path_a, path_b, jaccard))
            candidate_count += 1

            if len(insert_batch) >= 10_000:
                rc.executemany(
                    "INSERT INTO lsh_candidates (path_a, path_b, jaccard_est) VALUES (?,?,?)",
                    insert_batch
                )
                results_conn.commit()
                insert_batch.clear()

    if insert_batch:
        rc.executemany(
            "INSERT INTO lsh_candidates (path_a, path_b, jaccard_est) VALUES (?,?,?)",
            insert_batch
        )
        results_conn.commit()

    query_time = time.time() - t3
    elapsed = time.time() - t0

    print(f"\n  LSH complete in {elapsed:.1f}s")
    print(f"    Collect: {collect_time:.1f}s  |  Hash: {hash_time:.1f}s  |  Insert: {insert_time:.1f}s  |  Query: {query_time:.1f}s")
    print(f"  Candidate pairs: {candidate_count:,}")

    results_conn.close()

    stats = {
        "docs_indexed":    len(minhashes),
        "candidate_pairs": candidate_count,
        "threshold":       LSH_THRESHOLD,
        "workers":         n_workers,
        "elapsed_seconds": round(elapsed, 1),
    }

    state["completed_stages"].append("lsh")
    state["stats"]["lsh"] = stats
    save_state(state)
    print(f"  Results written -> {RESULTS_DB} (lsh_candidates table)")


# ---------------------------------------------------------------------------
# STAGE 3: Embeddings  (FAISS-accelerated)
# ---------------------------------------------------------------------------

def stage_embed():
    print("\n" + "="*60)
    print("STAGE 3: Embedding Similarity on LSH Candidates  (document level)")
    print("="*60)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ERROR: pip install sentence-transformers")
        sys.exit(1)

    try:
        import faiss
        HAS_FAISS = True
    except ImportError:
        HAS_FAISS = False
        print("  WARNING: faiss not installed, falling back to brute-force dot products")
        print("           pip install faiss-cpu  (or faiss-gpu) for faster search")

    state = load_state()
    if "embed" in state["completed_stages"]:
        print("  Already complete. Use --reset to re-run.")
        return

    results_conn = sqlite3.connect(RESULTS_DB)
    rc = results_conn.cursor()

    rc.execute("DELETE FROM embed_duplicates")
    results_conn.commit()

    rc.execute("SELECT path_a, path_b, jaccard_est FROM lsh_candidates ORDER BY jaccard_est DESC")
    candidates = rc.fetchall()

    if not candidates:
        print("  No LSH candidates found. Run --stage lsh first.")
        sys.exit(1)

    candidate_paths = set()
    for path_a, path_b, _ in candidates:
        candidate_paths.add(path_a)
        candidate_paths.add(path_b)

    print(f"  LSH candidates:       {len(candidates):,} pairs")
    print(f"  Unique docs to embed: {len(candidate_paths):,}")
    print(f"  Model: {EMBED_MODEL}")

    # Reassemble only the candidate documents
    source_conn = sqlite3.connect(SOURCE_DB)
    source_conn.execute("PRAGMA mmap_size=2147483648")
    sc = source_conn.cursor()
    sc.execute("SELECT filename, text FROM chunks ORDER BY filename")

    doc_chunks: dict[str, list] = {}

    print("  Streaming DB to collect candidate documents...")
    for filename, text in tqdm(sc, desc="  Scanning chunks"):
        base, sep, idx_str = filename.rpartition("_")
        if sep and idx_str.isdigit():
            base_key, chunk_idx = base, int(idx_str)
        else:
            base_key, chunk_idx = filename, 0

        if base_key not in candidate_paths:
            continue

        doc_chunks.setdefault(base_key, []).append((chunk_idx, text))

    source_conn.close()

    doc_texts = {p: reassemble_doc(chunks) for p, chunks in doc_chunks.items()}
    del doc_chunks  # free memory
    print(f"  Reassembled {len(doc_texts):,} documents")

    def truncate(text: str) -> str:
        words = text.split()
        return " ".join(words[:EMBED_TRUNCATE]) if len(words) > EMBED_TRUNCATE else text

    print(f"  Loading model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    ordered_paths = list(doc_texts.keys())
    ordered_texts = [truncate(doc_texts[p]) for p in ordered_paths]
    del doc_texts  # free the big dict before embedding allocations
    path_to_idx   = {p: i for i, p in enumerate(ordered_paths)}

    t0 = time.time()
    print(f"  Embedding {len(ordered_texts):,} documents (batch={EMBED_BATCH})...")

    embeddings = model.encode(
        ordered_texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    del ordered_texts  # free strings
    embed_time = time.time() - t0
    print(f"  Embedding complete in {embed_time:.1f}s")

    # Score candidate pairs
    # If candidate count is very large AND we have FAISS, we could do an ANN
    # search instead of iterating pairs. But for typical LSH output (thousands
    # to low millions of pairs), direct dot products are fine and exact.
    print("  Scoring candidate pairs...")
    insert_batch = []
    confirmed    = 0
    skipped      = 0

    for path_a, path_b, _ in tqdm(candidates, desc="  Scoring"):
        idx_a = path_to_idx.get(path_a)
        idx_b = path_to_idx.get(path_b)
        if idx_a is None or idx_b is None:
            skipped += 1
            continue
        sim = float(np.dot(embeddings[idx_a], embeddings[idx_b]))
        if sim >= EMBED_THRESHOLD:
            insert_batch.append((path_a, path_b, sim))
            confirmed += 1

            if len(insert_batch) >= 5_000:
                rc.executemany(
                    "INSERT INTO embed_duplicates (path_a, path_b, cosine_sim) VALUES (?,?,?)",
                    insert_batch
                )
                results_conn.commit()
                insert_batch.clear()

    if insert_batch:
        rc.executemany(
            "INSERT INTO embed_duplicates (path_a, path_b, cosine_sim) VALUES (?,?,?)",
            insert_batch
        )
        results_conn.commit()

    elapsed = time.time() - t0
    print(f"\n  Scoring complete in {elapsed:.1f}s")
    print(f"  Confirmed pairs (>={EMBED_THRESHOLD}): {confirmed:,} of {len(candidates):,}")
    if skipped:
        print(f"  Skipped (missing embeddings): {skipped:,}")

    results_conn.close()

    stats = {
        "lsh_candidates":   len(candidates),
        "docs_embedded":    len(ordered_paths),
        "confirmed_pairs":  confirmed,
        "threshold":        EMBED_THRESHOLD,
        "model":            EMBED_MODEL,
        "elapsed_seconds":  round(elapsed, 1),
    }

    state["completed_stages"].append("embed")
    state["stats"]["embed"] = stats
    save_state(state)
    print(f"  Results written -> {RESULTS_DB} (embed_duplicates table)")


# ---------------------------------------------------------------------------
# Status / Summary
# ---------------------------------------------------------------------------

def print_status():
    state = load_state()
    print("\n" + "="*60)
    print("Pipeline Status")
    print("="*60)

    stage_labels = {
        "hash":  "Stage 1: Hash Deduplication",
        "lsh":   "Stage 2: LSH / MinHash",
        "embed": "Stage 3: Embedding Similarity",
    }

    for s in ["hash", "lsh", "embed"]:
        done = s in state["completed_stages"]
        mark = "\u2713" if done else "\u2022"
        print(f"  [{mark}] {stage_labels[s]}")
        if done and s in state.get("stats", {}):
            for k, v in state["stats"][s].items():
                print(f"        {k}: {v}")

    if Path(RESULTS_DB).exists():
        conn = sqlite3.connect(RESULTS_DB)
        print()
        for table in ("hash_duplicates", "lsh_candidates", "embed_duplicates"):
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {n:,} rows")
            except Exception:
                pass

        # Summary: total unique documents flagged across all stages
        try:
            flagged = set()
            for row in conn.execute("SELECT DISTINCT base_path FROM hash_duplicates"):
                flagged.add(row[0])
            for row in conn.execute("SELECT path_a FROM embed_duplicates UNION SELECT path_b FROM embed_duplicates"):
                flagged.add(row[0])
            if flagged:
                print(f"\n  Total unique documents flagged as duplicates: {len(flagged):,}")
        except Exception:
            pass

        conn.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document-level dedup pipeline for Baytown_chunks.db"
    )
    parser.add_argument(
        "--stage",
        choices=["hash", "lsh", "embed", "status"],
        required=True,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear this stage's completion flag so it can be re-run",
    )
    args = parser.parse_args()

    if args.reset and args.stage != "status":
        state = load_state()
        if args.stage in state["completed_stages"]:
            state["completed_stages"].remove(args.stage)
            state.get("stats", {}).pop(args.stage, None)
            save_state(state)
            print(f"  Reset '{args.stage}' -- will re-run")

    dispatch = {
        "hash":   stage_hash,
        "lsh":    stage_lsh,
        "embed":  stage_embed,
        "status": print_status,
    }
    dispatch[args.stage]()