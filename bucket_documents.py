"""
bucket_documents.py - Classify documents into buckets using a local LLM.

Reads the chunk database, sends the first N tokens of each document to Ollama
for classification, and creates Windows .lnk shortcuts organized by bucket.

Usage:
    python bucket_documents.py                     # uses bucket_config.json
    python bucket_documents.py my_config.json      # custom config path

Requires:
    - Ollama running locally (ollama serve)
    - Model pulled (default: ollama pull qwen2.5:7b)
    - Existing chunk database from document ingestion
"""

import json
import os
import re
import sqlite3
import struct
import sys
import time

import requests
from tqdm import tqdm


# ============================================================================
# Bucket definitions
# ============================================================================

BUCKET_NAMES = {
    0:  "00_Unclassified",
    1:  "01_PreCon_Contracts",
    2:  "02_PreCon_Proposals",
    3:  "03_Design_Correspondence",
    4:  "04_Design_Drawings",
    5:  "05_Design_Geotech",
    6:  "06_Design_Specifications",
    7:  "07_Design_Submittals",
    8:  "08_CNST_AsBuilt_Survey",
    9:  "09_CNST_Certificates",
    10: "10_CNST_Change_Orders",
    11: "11_CNST_Correspondence",
    12: "12_CNST_Field_Reports",
    13: "13_CNST_Meeting_Minutes",
    14: "14_CNST_Payment_Applications",
    15: "15_CNST_Photos",
    16: "16_CNST_PNA_Design_Inputs",
    17: "17_CNST_Product_Data",
    18: "18_CNST_RFIs",
    19: "19_CNST_Testing_Reports",
    20: "20_CNST_Warranties",
    21: "21_PostCNST_Correspondence",
}

# NOTE: The system prompt says "integer 0-20" but bucket 21 exists.
# The code handles 0-21; the LLM will still produce 21 when appropriate.
SYSTEM_PROMPT = r"""You are a document classification system for a construction project. You will receive the first few pages of text extracted from a document. Your job is to assign the document to exactly one bucket and generate a short descriptive filename.

RESPOND WITH EXACTLY ONE LINE IN THIS FORMAT:
<bucket_number> <descriptive_name>

- bucket_number: integer 0-20
- descriptive_name: max 20 characters, no spaces, use underscores, filesystem-safe characters only (no / \ : * ? " < > |)
- The descriptive name should summarize the document content (not the bucket), e.g. "SoilBoring_Rpt_2019" or "CO4_HVAC_Ductwork"

Do not output anything else. No explanation, no preamble, no formatting.

BUCKETS:

0  - Unclassified (use ONLY when the document clearly does not fit any bucket below, or the text is too ambiguous/corrupt to classify)
1  - PreCon - Contracts (executed agreements, contract documents, amendments, terms and conditions between parties)
2  - PreCon - Proposals (bids, proposals, scope of work offers, fee proposals, qualifications submittals prior to contract)
3  - Design - Correspondence (letters, emails, memos, transmittals related to the design phase)
4  - Design - Drawings (drawing lists, drawing indexes, references to sheet numbers, plan/section/detail annotations, title block text)
5  - Design - Geotech (geotechnical reports, soil boring logs, subsurface investigations, foundation recommendations)
6  - Design - Specifications (project specifications, technical spec sections, CSI-formatted divisions, material/performance requirements)
7  - Design - Submittals (shop drawings, product data, material samples, submittal transmittals and review logs during design)
8  - CNST - As-Built Survey (as-built survey data, field survey coordinates, red-line as-built markups, final survey plats)
9  - CNST - Certificates (certificates of occupancy, substantial completion, insurance certificates, lien waivers, compliance certifications)
10 - CNST - Change Orders (change order proposals, approved change orders, COP/COR documentation, cost/schedule impact of changes)
11 - CNST - Correspondence (letters, emails, memos, transmittals, notices during the construction phase)
12 - CNST - Field Reports (daily field reports, inspection logs, site observation reports, field notes from site visits)
13 - CNST - Meeting Minutes (meeting minutes, progress meeting notes, OAC meeting records, pre-construction meeting minutes)
14 - CNST - Payment Applications (pay applications, schedule of values, AIA G702/G703 forms, invoice backup, lien releases tied to payment)
15 - CNST - Photos (photo logs, site photograph transmittals, progress photo documentation, image-reference sheets)
16 - CNST - PNA Recommended Design Inputs (documents from ________ containing recommended design parameters, criteria, or inputs)
17 - CNST - Product Data (product cut sheets, material data sheets, manufacturer literature, catalog pages, technical data submittals during construction)
18 - CNST - RFIs (requests for information, RFI logs, RFI responses, clarification requests during construction)
19 - CNST - Testing Reports (material testing reports, compressive strength tests, soil compaction tests, special inspection reports, lab results)
20 - CNST - Warranties (warranty documents, guarantees, extended warranty certificates, maintenance bonds)
21 - Post-CNST - Correspondence (letters, emails, memos related to post-construction phase, closeout correspondence, warranty-period communications)

CLASSIFICATION GUIDANCE:

Phase takes priority: If a letter is about design issues, it goes in 3 (Design - Correspondence), not 11 (CNST - Correspondence). Look for date context, project phase references, and involved parties to determine phase.

Submittals vs Product Data: Bucket 7 is for submittals during design. Bucket 17 is for product data submitted during construction. If the document is a construction-phase submittal containing product/material data, prefer 17.

Drawings: Documents in bucket 4 will often have minimal prose. Look for drawing numbers (e.g. S-101, A-201, M-001), sheet indexes, or references to plan/section/detail views.

Photos: With text-only input, look for cues such as "photo log", "site photographs", "progress photos", image file references, camera metadata, or transmittal cover sheets referencing attached photographs.

Change Orders vs RFIs: A change order modifies cost or scope. An RFI requests clarification without necessarily changing scope. If an RFI leads to a change, classify based on what the document itself is, not what it led to.

When in doubt between two plausible buckets, choose the more specific one."""


# ============================================================================
# Config
# ============================================================================

CONFIG_DEFAULTS = {
    "chunk_db": "BAYT_PROD_chunks.db",
    "output_dir": "./bucketed_output",
    "progress_db": "bucket_progress.db",
    "ollama_url": "http://localhost:11434/api/generate",
    "model": "qwen2.5:7b",
    "token_budget": 500,
    "temperature": 0.1,
    "context_window": 8192,
}


def load_config(path):
    """Load JSON config, filling in defaults for any missing keys."""
    with open(path) as f:
        cfg = json.load(f)
    for key, default in CONFIG_DEFAULTS.items():
        cfg.setdefault(key, default)
    return cfg


# ============================================================================
# Chunk database helpers
# ============================================================================

def _page_sort_key(chunk_filename):
    """Extract page number from 'path/to/file.pdf_3' for sorting."""
    _, sep, page_str = chunk_filename.rpartition("_")
    return int(page_str) if sep and page_str.isdigit() else 0


def get_unique_documents(db_path):
    """
    Return {document_path: [chunk_filenames, ...]} grouped by source document.

    Chunk filenames look like 'I:/.../file.pdf_0', 'I:/.../file.pdf_1', etc.
    We strip the '_N' page suffix to recover the original document path.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT filename FROM chunks ORDER BY filename").fetchall()
    conn.close()

    docs = {}
    for (filename,) in rows:
        base, sep, page_str = filename.rpartition("_")
        doc_path = base if (sep and page_str.isdigit()) else filename
        docs.setdefault(doc_path, []).append(filename)

    return docs


def get_document_text(db_path, chunk_filenames, token_budget):
    """
    Concatenate page text in page order up to token_budget tokens, then clip.

    Token estimate: 1 token ~ 4 characters (matches create_chunk_database.py).
    """
    chunk_filenames_sorted = sorted(chunk_filenames, key=_page_sort_key)
    char_budget = token_budget * 4

    conn = sqlite3.connect(db_path)
    text = ""

    for filename in chunk_filenames_sorted:
        row = conn.execute(
            "SELECT text FROM chunks WHERE filename = ?", (filename,)
        ).fetchone()
        if row and row[0]:
            text += row[0] + "\n"
            if len(text) >= char_budget:
                text = text[:char_budget]
                # Clip at last space so we don't cut mid-word
                last_space = text.rfind(" ")
                if last_space > char_budget * 0.8:
                    text = text[:last_space]
                break

    conn.close()
    return text.strip()


# ============================================================================
# LLM interaction
# ============================================================================

def classify_document(text, config):
    """Send document text to Ollama and return the raw response string."""
    payload = {
        "model": config["model"],
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "options": {
            "temperature": config["temperature"],
            "num_ctx": config["context_window"],
        },
    }

    resp = requests.post(config["ollama_url"], json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def parse_classification(raw):
    """
    Parse '<bucket_number> <descriptive_name>' from LLM response.

    Returns (bucket, sanitized_name).  Falls back to (0, 'parse_error').
    """
    # Take first non-empty line (models sometimes add stray whitespace)
    for line in raw.splitlines():
        line = line.strip()
        if line:
            raw = line
            break

    match = re.match(r"^(\d+)\s+(\S+)", raw)
    if not match:
        return 0, "parse_error"

    bucket = int(match.group(1))
    name = match.group(2)

    if bucket not in BUCKET_NAMES:
        bucket = 0

    # Sanitize: filesystem-safe chars only, max 20 characters
    name = re.sub(r'[^\w\-]', "_", name)[:20]
    if not name:
        name = "unnamed"

    return bucket, name


# ============================================================================
# Windows .lnk shortcut creation
# ============================================================================

def create_lnk(target_path, lnk_path):
    """
    Create a Windows .lnk shortcut pointing at target_path.

    Tries win32com (full-featured, Windows only) first, then falls back to
    writing a minimal binary .lnk per the MS-SHLLINK specification.
    """
    try:
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(lnk_path))
        shortcut.Targetpath = str(target_path)
        shortcut.save()
        return
    except ImportError:
        pass

    _write_lnk_binary(str(target_path), str(lnk_path))


def _write_lnk_binary(target_path, lnk_path):
    """
    Write a minimal Shell Link Binary (.lnk) file.

    Implements just enough of the MS-SHLLINK spec for Windows Explorer
    to resolve the target path.  No icon, no arguments, no working dir.
    """
    target_bytes = target_path.encode("ascii", errors="replace") + b"\x00"
    suffix_bytes = b"\x00"  # empty CommonPathSuffix

    # ── VolumeID (17 bytes) ─────────────────────────────────────────────
    #  VolumeIDSize(4) + DriveType(4) + Serial(4) + LabelOffset(4) + label(\0)
    volume_id = struct.pack("<IIII", 17, 3, 0, 0x10) + b"\x00"

    # ── LinkInfo ────────────────────────────────────────────────────────
    li_header_size = 0x1C  # 28 bytes (mandatory fields)
    vol_id_offset = li_header_size
    base_path_offset = vol_id_offset + len(volume_id)
    suffix_offset = base_path_offset + len(target_bytes)
    li_size = suffix_offset + len(suffix_bytes)

    link_info = struct.pack(
        "<IIIIIII",
        li_size,
        li_header_size,
        0x01,               # Flags: VolumeIDAndLocalBasePath
        vol_id_offset,
        base_path_offset,
        0,                   # CommonNetworkRelativeLinkOffset (unused)
        suffix_offset,
    ) + volume_id + target_bytes + suffix_bytes

    # ── ShellLinkHeader (76 bytes) ──────────────────────────────────────
    header = struct.pack("<I", 0x4C)                             # HeaderSize
    header += bytes([                                            # LinkCLSID
        0x01, 0x14, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46,
    ])
    header += struct.pack("<I", 0x00000002)                      # LinkFlags: HasLinkInfo
    header += struct.pack("<I", 0x00000000)                      # FileAttributes
    header += b"\x00" * 24                                       # Creation/Access/Write times
    header += struct.pack("<III", 0, 0, 1)                       # FileSize, IconIndex, ShowCommand
    header += struct.pack("<HH", 0, 0)                           # HotKey, Reserved1
    header += struct.pack("<II", 0, 0)                           # Reserved2, Reserved3

    os.makedirs(os.path.dirname(lnk_path), exist_ok=True)
    with open(lnk_path, "wb") as f:
        f.write(header)
        f.write(link_info)


# ============================================================================
# Progress tracking (SQLite)
# ============================================================================

def init_progress_db(db_path):
    """Create the progress table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bucketed (
            document_path  TEXT PRIMARY KEY,
            bucket         INTEGER NOT NULL,
            short_name     TEXT NOT NULL,
            raw_response   TEXT,
            processed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_processed_set(db_path):
    """Return set of document paths already classified."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT document_path FROM bucketed").fetchall()
    conn.close()
    return {r[0] for r in rows}


def log_result(db_path, doc_path, bucket, name, raw_response):
    """Record a classification result."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO bucketed VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (doc_path, bucket, name, raw_response),
    )
    conn.commit()
    conn.close()


# ============================================================================
# Main
# ============================================================================

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "bucket_config.json"

    if not os.path.exists(config_path):
        print(f"ERROR: Config file '{config_path}' not found.")
        return 1

    config = load_config(config_path)

    chunk_db = config["chunk_db"]
    output_dir = config["output_dir"]
    progress_db = config["progress_db"]
    token_budget = config["token_budget"]

    # ── Validate chunk DB ───────────────────────────────────────────────
    if not os.path.exists(chunk_db):
        print(f"ERROR: Chunk database '{chunk_db}' not found.")
        return 1

    # ── Check Ollama connectivity ───────────────────────────────────────
    ollama_base = config["ollama_url"].rsplit("/", 1)[0]
    try:
        requests.get(f"{ollama_base}/api/tags", timeout=5)
    except requests.exceptions.RequestException:
        print("WARNING: Cannot reach Ollama. Will fail on first classify call.")

    # ── Init progress DB and bucket folders ─────────────────────────────
    init_progress_db(progress_db)
    processed = get_processed_set(progress_db)

    for folder_name in BUCKET_NAMES.values():
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)

    # ── Discover documents ──────────────────────────────────────────────
    documents = get_unique_documents(chunk_db)
    to_process = {k: v for k, v in documents.items() if k not in processed}

    print(f"Documents in chunk DB : {len(documents)}")
    print(f"Already bucketed      : {len(processed)}")
    print(f"Remaining             : {len(to_process)}")
    print(f"Model: {config['model']}   Token budget: {token_budget}")
    print("-" * 60)

    if not to_process:
        print("Nothing to do.")
        return 0

    # ── Classify each document ──────────────────────────────────────────
    error_count = 0

    for doc_path, chunk_files in tqdm(to_process.items(), desc="Classifying"):
        try:
            text = get_document_text(chunk_db, chunk_files, token_budget)

            if not text:
                bucket, name, raw = 0, "empty_document", ""
            else:
                raw = classify_document(text, config)
                bucket, name = parse_classification(raw)

            # Build .lnk path; handle name collisions with a counter
            folder = BUCKET_NAMES[bucket]
            lnk_path = os.path.join(output_dir, folder, f"{name}.lnk")
            counter = 1
            while os.path.exists(lnk_path):
                lnk_path = os.path.join(output_dir, folder, f"{name}_{counter}.lnk")
                counter += 1

            create_lnk(doc_path, lnk_path)
            log_result(progress_db, doc_path, bucket, name, raw)

        except Exception as e:
            # Don't log to progress DB so it retries on next run
            print(f"\nERROR on {doc_path}: {e}")
            error_count += 1

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Classification summary")
    print("=" * 60)

    conn = sqlite3.connect(progress_db)
    for bucket_id, folder in sorted(BUCKET_NAMES.items()):
        count = conn.execute(
            "SELECT COUNT(*) FROM bucketed WHERE bucket = ?", (bucket_id,)
        ).fetchone()[0]
        if count:
            print(f"  {folder}: {count}")
    conn.close()

    if error_count:
        print(f"\n  {error_count} errors (not logged — will retry next run)")

    return 0


if __name__ == "__main__":
    exit(main())
