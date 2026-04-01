import sqlite3
from tqdm import tqdm
from pathlib import Path

basePath = "I:/BOS/Projects/2023/230903.00-BAYT/APBPXXX/APBP Production Export"

CHUNK_LINES   = 500   # lines per chunk
OVERLAP_LINES = 50    # lines of overlap between chunks

# ---------------------------------------------------------------------------
# Step 1: Find files
# ---------------------------------------------------------------------------
print("Scanning for TXT files...")
base = Path(basePath)
files = []

for i, p in enumerate(base.glob("*/TEXT/*/*.txt"), 1):
    files.append(str(p))
    if i % 100 == 0:
        print(f"\r  {i} files...", end="", flush=True)

print(f"\r  Done — found {len(files)} files")


OUTPUT_DB = "Baytown_chunks.db"

# ---------------------------------------------------------------------------
# Step 2: Database setup
# ---------------------------------------------------------------------------
print(f"Opening database: {OUTPUT_DB}")
conn = sqlite3.connect(OUTPUT_DB)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        filename    TEXT NOT NULL,
        text        TEXT NOT NULL,
        token_count INTEGER,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON chunks(filename)')
conn.commit()
print("  Database ready")

# ---------------------------------------------------------------------------
# Step 3: Check what's already been parsed
# ---------------------------------------------------------------------------
print("Checking for previously parsed files...")
cursor.execute('SELECT DISTINCT filename FROM chunks')
existing_entries = set(row[0] for row in cursor.fetchall())

# filename in DB looks like "I:/.../foo.txt_0"
already_parsed_files = set()
for entry in existing_entries:
    base_part, sep, chunk_str = entry.rpartition('_')
    if sep and chunk_str.isdigit():
        already_parsed_files.add(base_part)

files_to_process = [f for f in files if f not in already_parsed_files]
print(f"  {len(existing_entries)} chunk rows from {len(already_parsed_files)} files already in DB")

print(f"\nTotal files found:    {len(files)}")
print(f"Already in database:  {len(files) - len(files_to_process)}")
print(f"Remaining to process: {len(files_to_process)}")

# ---------------------------------------------------------------------------
# Helper: read a .txt file with encoding fallback
# ---------------------------------------------------------------------------
def read_txt(path: str) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {path} with any attempted encoding")

# ---------------------------------------------------------------------------
# Helper: split lines into overlapping chunks
# ---------------------------------------------------------------------------
def line_chunks(text: str, chunk_size: int, overlap: int):
    lines = text.splitlines()
    if not lines:
        return
    step = max(1, chunk_size - overlap)
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i : i + chunk_size])
        if chunk.strip():
            yield i // step, chunk

# ---------------------------------------------------------------------------
# Process files
# ---------------------------------------------------------------------------
error_count = 0

for file in tqdm(files_to_process, desc="Processing TXTs"):
    try:
        text = read_txt(file)
        for chunk_idx, chunk_text in line_chunks(text, CHUNK_LINES, OVERLAP_LINES):
            token_count = len(chunk_text) // 4
            cursor.execute(
                'INSERT INTO chunks (filename, text, token_count) VALUES (?, ?, ?)',
                (f"{file}_{chunk_idx}", chunk_text, token_count)
            )
        conn.commit()

    except Exception as e:
        print(f"\nERROR processing {file}: {e}")
        conn.rollback()
        error_count += 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cursor.execute('SELECT COUNT(*) FROM chunks')
total_chunks = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(DISTINCT filename) FROM chunks')
total_entries = cursor.fetchone()[0]

print(f"\nDone.  {total_chunks} total chunks across {total_entries} chunk entries in {OUTPUT_DB}")
if error_count:
    print(f"  ⚠️  {error_count} files failed — re-run to retry them")

conn.close()