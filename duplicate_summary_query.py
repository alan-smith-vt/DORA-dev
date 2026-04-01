"""Quick stats on LSH candidate pairs in dedup_results.db"""

import sqlite3

conn = sqlite3.connect("data/dedup_results.db")

unique_docs = conn.execute("""
    SELECT COUNT(DISTINCT doc) FROM (
        SELECT path_a AS doc FROM lsh_candidates
        UNION
        SELECT path_b AS doc FROM lsh_candidates
    )
""").fetchone()[0]

total_pairs = conn.execute("SELECT COUNT(*) FROM lsh_candidates").fetchone()[0]

print(f"  Total LSH candidate pairs:  {total_pairs:,}")
print(f"  Unique documents in pairs:  {unique_docs:,}")
print(f"  (out of 53,201 unique contents from hash stage)")

# Top 20 most-connected documents (appear in the most pairs)
print(f"\n  Top 20 most-connected documents:")
rows = conn.execute("""
    SELECT doc, COUNT(*) as pair_count FROM (
        SELECT path_a AS doc FROM lsh_candidates
        UNION ALL
        SELECT path_b AS doc FROM lsh_candidates
    )
    GROUP BY doc
    ORDER BY pair_count DESC
    LIMIT 20
""").fetchall()

for path, count in rows:
    print(f"    {count:>6,} pairs  {path}")

# Distribution: how many docs have N pairs
print(f"\n  Pair count distribution:")
buckets = conn.execute("""
    WITH doc_counts AS (
        SELECT doc, COUNT(*) as n FROM (
            SELECT path_a AS doc FROM lsh_candidates
            UNION ALL
            SELECT path_b AS doc FROM lsh_candidates
        )
        GROUP BY doc
    )
    SELECT
        CASE
            WHEN n = 1 THEN '1'
            WHEN n BETWEEN 2 AND 5 THEN '2-5'
            WHEN n BETWEEN 6 AND 20 THEN '6-20'
            WHEN n BETWEEN 21 AND 100 THEN '21-100'
            WHEN n BETWEEN 101 AND 500 THEN '101-500'
            ELSE '500+'
        END as bucket,
        COUNT(*) as num_docs,
        MIN(n) as min_pairs,
        MAX(n) as max_pairs
    FROM doc_counts
    GROUP BY bucket
    ORDER BY min_pairs
""").fetchall()

for bucket, num_docs, mn, mx in buckets:
    print(f"    {bucket:>7} pairs: {num_docs:,} documents")

conn.close()