import os
import sqlite3
import time
from glob import glob
from tqdm import tqdm
import pdfplumber

#basePath = "I:/BOS/Projects/2025/250836.00-EXTN/_Working  Folder/Correspondence_Emails/"
basePath = "I:/BOS/Projects/2023/230903.00-BAYT/APBPXXX/20260319_Incoming Prod Export Searchable PDFs/20260319_Export Searchable PDFs/IMAGES/IMG001"

# ---------------------------------------------------------------------------
# Step 1: Find files
# ---------------------------------------------------------------------------
print("Scanning for PDF files...")
t0 = time.time()
files = glob(basePath + "/*.pdf")
print(f"  Found {len(files)} files  ({time.time() - t0:.1f}s)")

OUTPUT_DB = "BAYT_PROD_chunks.db"

# ---------------------------------------------------------------------------
# Step 2: Sort by filesize (stat calls over network can be slow)
# ---------------------------------------------------------------------------
print("Reading file sizes for sort...")
t0 = time.time()
file_sizes = {}
for f in tqdm(files, desc="Stat files", unit="file"):
	try:
		file_sizes[f] = os.path.getsize(f)
	except OSError:
		file_sizes[f] = 0  # put inaccessible files first, they'll error out quickly
files.sort(key=lambda f: file_sizes[f])
total_size_mb = sum(file_sizes.values()) / (1024 * 1024)
print(f"  Sorted.  Total size: {total_size_mb:.1f} MB  ({time.time() - t0:.1f}s)")

# ---------------------------------------------------------------------------
# Step 3: Database setup — open or create, never delete existing
# ---------------------------------------------------------------------------
print(f"Opening database: {OUTPUT_DB}")
t0 = time.time()
db_is_new = not os.path.exists(OUTPUT_DB)
conn = sqlite3.connect(OUTPUT_DB)
cursor = conn.cursor()

cursor.execute('''
	CREATE TABLE IF NOT EXISTS chunks (
		chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
		filename TEXT NOT NULL,
		text TEXT NOT NULL,
		token_count INTEGER,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON chunks(filename)')
conn.commit()
print(f"  {'Created new' if db_is_new else 'Opened existing'} database  ({time.time() - t0:.1f}s)")

# ---------------------------------------------------------------------------
# Step 4: Check what's already been parsed
# ---------------------------------------------------------------------------
print("Checking for previously parsed files...")
t0 = time.time()
cursor.execute('SELECT DISTINCT filename FROM chunks')
existing_entries = set(row[0] for row in cursor.fetchall())

# Derive the set of source files that already have at least one page stored.
# filename in the DB looks like "I:/.../foo.pdf_0", so rsplit once on '_'.
already_parsed_files = set()
for entry in existing_entries:
	base, sep, page_str = entry.rpartition('_')
	if sep and page_str.isdigit():
		already_parsed_files.add(base)

files_to_process = [f for f in files if f not in already_parsed_files]
print(f"  {len(existing_entries)} chunk rows from {len(already_parsed_files)} files already in DB  ({time.time() - t0:.1f}s)")

print(f"\nTotal files found:    {len(files)}")
print(f"Already in database:  {len(files) - len(files_to_process)}")
print(f"Remaining to process: {len(files_to_process)}")

# ---------------------------------------------------------------------------
# Process files — commit after each file so crashes lose at most one file
# ---------------------------------------------------------------------------
error_count = 0

for file in tqdm(files_to_process, desc="Processing PDFs"):
	try:
		with pdfplumber.open(file) as reader:
			for i, page in enumerate(reader.pages):
				content = page.extract_text()
				if not content:
					continue
				token_count = len(content) // 4  # rough estimate
				cursor.execute(
					'INSERT INTO chunks (filename, text, token_count) VALUES (?, ?, ?)',
					(file + f"_{i}", content, token_count)
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

print(f"\nDone.  {total_chunks} total chunks across {total_entries} page entries in {OUTPUT_DB}")
if error_count:
	print(f"  ⚠️  {error_count} files failed — re-run to retry them")

conn.close()