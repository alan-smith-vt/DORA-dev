import sqlite3
conn = sqlite3.connect("BAYT_PROD_chunks.db")
cur = conn.cursor()

# Check before
cur.execute("SELECT filename FROM chunks WHERE filename LIKE '%PNA_00002130%'")
before = cur.fetchall()
print(f"Matches BEFORE delete: {len(before)}")
for row in before:
    print(f"  {row[0]}")

# Attempt delete
cur.execute("DELETE FROM chunks WHERE filename LIKE '%PNA_00002130%'")
conn.commit()


# Check after
cur.execute("SELECT filename FROM chunks WHERE filename LIKE '%PNA_00002130%'")
after = cur.fetchall()
print(f"Matches AFTER delete: {len(after)}")
for row in after:
    print(f"  {row[0]}")
conn.close()