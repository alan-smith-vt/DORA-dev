"""
normalize_entities.py
=====================
Phase 1 entity deduplication: deterministic normalization.

Reads a DORA entity database, normalizes entity names (case folding,
whitespace collapse, title/suffix stripping, punctuation cleanup),
groups entities that share the same normalized key **across types**,
picks a canonical representative per group, and writes a new database
with merged entities and updated relationships.

Also produces an HTML inspection report so you can eyeball the merges
before trusting them downstream.

Usage:
    python normalize_entities.py                           # defaults
    python normalize_entities.py --input my_entities.db    # custom input
    python normalize_entities.py --input my.db --output clean.db --report report.html

Inputs:
    - Source entity database (SQLite, schema from extract_entities.py)

Outputs:
    - New entity database with merged entities and rewritten relationships
    - HTML report showing all merge groups and before/after statistics
"""

import argparse
import json
import re
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalization rules
# ---------------------------------------------------------------------------

# Titles to strip (case-insensitive, with optional trailing period)
TITLE_PREFIXES = [
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "sir", "dame", "rev", "reverend", "hon", "honorable",
    "sgt", "sargent", "cpl", "corporal", "pvt", "private",
    "lt", "lieutenant", "cpt", "captain", "maj", "major",
    "col", "colonel", "gen", "general", "adm", "admiral",
    "eng", "engr", "atty", "esq",
]

# Organization suffixes to strip
ORG_SUFFIXES = [
    "inc", "incorporated",
    "llc", "l.l.c",
    "ltd", "limited",
    "corp", "corporation",
    "co", "company",
    "plc",
    "lp", "l.p",
    "llp", "l.l.p",
    "pllc",
    "sa", "s.a",
    "gmbh",
    "ag",
    "pty",
    "na", "n.a",
]


def normalize_name(name: str) -> str:
    """
    Produce a normalized key from an entity name.

    Steps:
      1. Strip and case fold
      2. Remove content in parentheses (often role annotations)
      3. Strip leading titles (Mr., Dr., etc.)
      4. Strip trailing org suffixes (Inc., LLC, etc.)
      5. Collapse abbreviation dots (U.S.A. -> USA, J. -> J)
      6. Remove remaining non-alphanumeric chars except spaces
      7. Collapse whitespace
    """
    if not name:
        return ""

    s = name.strip().lower()

    # Remove parenthetical annotations: "John Doe (Site Inspector)" -> "John Doe"
    s = re.sub(r'\s*\(.*?\)\s*', ' ', s)

    # Strip trailing comma + anything (often "Doe, John" isn't an issue but
    # "ACME Corp, Inc." is — we handle via suffix removal below)

    # Strip leading titles
    for title in TITLE_PREFIXES:
        pattern = rf'^{re.escape(title)}\.?\s+'
        s = re.sub(pattern, '', s)

    # Strip trailing suffixes (with optional leading comma)
    for suffix in ORG_SUFFIXES:
        pattern = rf'[,\s]+{re.escape(suffix)}\.?\s*$'
        s = re.sub(pattern, '', s)

    # Collapse abbreviation dots: "U.S.A." -> "usa", "J. Smith" -> "j smith"
    # Only collapse dots that are between single letters or at word end after single letter
    s = re.sub(r'(?<!\w)(\w)\.(?=\s|$|\w\.)', r'\1', s)
    # Remaining trailing dots
    s = re.sub(r'\.\s*$', '', s)

    # Remove non-alphanumeric except spaces and hyphens
    s = re.sub(r'[^a-z0-9\s\-]', '', s)

    # Collapse hyphens surrounded by spaces, and multiple spaces
    s = re.sub(r'\s*-\s*', '-', s)
    s = re.sub(r'\s+', ' ', s).strip()

    return s


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def build_merge_groups(entities):
    """
    Group entities by normalized name key (across all types).

    Args:
        entities: list of dicts with keys: entity_id, name, type, description,
                  first_seen_document, documents_appeared

    Returns:
        groups: dict mapping normalized_key -> list of entity dicts
        name_map: dict mapping original_name -> canonical_name
    """
    key_groups = defaultdict(list)

    for ent in entities:
        key = normalize_name(ent["name"])
        if not key:
            key = ent["name"].strip().lower()  # fallback: at least lowercase
        key_groups[key].append(ent)

    name_map = {}  # original name -> canonical name
    merge_groups = []  # only groups with >1 member

    for key, group in key_groups.items():
        # Pick canonical: most total document appearances, then longest original name
        def sort_key(e):
            try:
                doc_count = len(json.loads(e["documents_appeared"])) if e["documents_appeared"] else 0
            except (json.JSONDecodeError, TypeError):
                doc_count = 0
            return (-doc_count, -len(e["name"]))

        group.sort(key=sort_key)
        canonical = group[0]

        for ent in group:
            name_map[ent["name"]] = canonical["name"]

        if len(group) > 1:
            merge_groups.append({
                "canonical": canonical,
                "members": group,
                "normalized_key": key,
            })

    return merge_groups, name_map


# ---------------------------------------------------------------------------
# DB rewrite
# ---------------------------------------------------------------------------

def rewrite_database(source_db: str, output_db: str, name_map: dict, merge_groups: list):
    """
    Copy source DB to output DB, then:
      1. Merge entity rows that map to the same canonical name
      2. Rewrite relationship source/target
      3. Deduplicate resulting relationships
      4. Update document sender/recipient/cc/bcc
    """
    # Start with a full copy
    shutil.copy2(source_db, output_db)

    conn = sqlite3.connect(output_db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # --- 1. Merge entities ---------------------------------------------------
    # Build a map: canonical_name -> merged entity info
    canonical_info = {}
    for mg in merge_groups:
        canon_name = mg["canonical"]["name"]
        all_docs = set()
        best_desc = ""
        best_desc_len = 0
        best_type = mg["canonical"]["type"]
        best_type_docs = 0
        first_seen = mg["canonical"]["first_seen_document"]

        for ent in mg["members"]:
            # Merge documents_appeared
            try:
                docs = json.loads(ent["documents_appeared"]) if ent["documents_appeared"] else []
            except (json.JSONDecodeError, TypeError):
                docs = []
            all_docs.update(docs)

            # Best description: longest
            desc = ent["description"] or ""
            if len(desc) > best_desc_len:
                best_desc = desc
                best_desc_len = len(desc)

            # Best type: from entity with most docs
            doc_count = len(docs)
            if doc_count > best_type_docs:
                best_type_docs = doc_count
                best_type = ent["type"]

        canonical_info[canon_name] = {
            "type": best_type,
            "description": best_desc,
            "first_seen_document": first_seen,
            "documents_appeared": json.dumps(sorted(all_docs, key=str)),
        }

    # Delete non-canonical entity rows, update canonical rows
    for mg in merge_groups:
        canon_name = mg["canonical"]["name"]
        info = canonical_info[canon_name]

        # Delete all members (including canonical — we'll re-insert/update)
        member_ids = [ent["entity_id"] for ent in mg["members"]]
        placeholders = ",".join("?" * len(member_ids))
        c.execute(f"DELETE FROM entities WHERE entity_id IN ({placeholders})", member_ids)

        # Insert the merged canonical entity
        c.execute("""
            INSERT INTO entities (name, type, description, first_seen_document, documents_appeared)
            VALUES (?, ?, ?, ?, ?)
        """, (
            canon_name,
            info["type"],
            info["description"],
            info["first_seen_document"],
            info["documents_appeared"],
        ))

    conn.commit()

    # --- 2. Rewrite relationships ---------------------------------------------
    # Fetch all relationships, rewrite source/target
    c.execute("SELECT relationship_id, source, target FROM relationships")
    updates = []
    for row in c.fetchall():
        rid = row["relationship_id"]
        old_src = row["source"]
        old_tgt = row["target"]
        new_src = name_map.get(old_src, old_src)
        new_tgt = name_map.get(old_tgt, old_tgt)
        if new_src != old_src or new_tgt != old_tgt:
            updates.append((new_src, new_tgt, rid))

    if updates:
        c.executemany(
            "UPDATE relationships SET source = ?, target = ? WHERE relationship_id = ?",
            updates,
        )
    conn.commit()

    # --- 3. Deduplicate relationships -----------------------------------------
    # After rewriting, (source, target, type, document_id) may have duplicates.
    # Keep the one with the longest description.
    c.execute("""
        DELETE FROM relationships
        WHERE relationship_id NOT IN (
            SELECT MIN(relationship_id)
            FROM relationships
            GROUP BY source, target, type, document_id
        )
    """)
    rels_deduped = c.rowcount
    conn.commit()

    # --- 4. Update documents table sender/recipient/cc/bcc -------------------
    for field in ("sender", "recipient", "cc", "bcc"):
        c.execute(f"SELECT document_id, {field} FROM documents WHERE {field} IS NOT NULL AND {field} != ''")
        doc_updates = []
        for row in c.fetchall():
            old_val = row[field]
            new_val = name_map.get(old_val, old_val)
            if new_val != old_val:
                doc_updates.append((new_val, row["document_id"]))
        if doc_updates:
            c.executemany(
                f"UPDATE documents SET {field} = ? WHERE document_id = ?",
                doc_updates,
            )
    conn.commit()

    # --- Collect stats --------------------------------------------------------
    c.execute("SELECT COUNT(*) FROM entities")
    final_entity_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM relationships")
    final_rel_count = c.fetchone()[0]

    conn.close()

    return {
        "relationships_rewritten": len(updates),
        "relationships_deduped": rels_deduped,
        "final_entity_count": final_entity_count,
        "final_rel_count": final_rel_count,
    }


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_report(
    merge_groups, name_map, source_db, output_db,
    original_entity_count, original_rel_count,
    rewrite_stats, elapsed,
):
    """Generate a self-contained HTML report for inspecting merge decisions."""

    # Prepare merge group data for the report
    groups_json = []
    for mg in sorted(merge_groups, key=lambda g: -len(g["members"])):
        members = []
        for ent in mg["members"]:
            try:
                doc_count = len(json.loads(ent["documents_appeared"])) if ent["documents_appeared"] else 0
            except (json.JSONDecodeError, TypeError):
                doc_count = 0
            members.append({
                "name": ent["name"],
                "type": ent["type"],
                "description": ent["description"] or "",
                "doc_count": doc_count,
                "is_canonical": ent["name"] == mg["canonical"]["name"],
            })
        groups_json.append({
            "canonical": mg["canonical"]["name"],
            "normalized_key": mg["normalized_key"],
            "member_count": len(mg["members"]),
            "members": members,
        })

    stats = {
        "source_db": source_db,
        "output_db": output_db,
        "original_entities": original_entity_count,
        "final_entities": rewrite_stats["final_entity_count"],
        "entities_removed": original_entity_count - rewrite_stats["final_entity_count"],
        "original_relationships": original_rel_count,
        "final_relationships": rewrite_stats["final_rel_count"],
        "relationships_rewritten": rewrite_stats["relationships_rewritten"],
        "relationships_deduped": rewrite_stats["relationships_deduped"],
        "merge_groups": len(merge_groups),
        "entities_involved": sum(len(g["members"]) for g in merge_groups),
        "elapsed_seconds": round(elapsed, 2),
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entity Normalization Report</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --canon-bg: #1a2332;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.5; padding: 2rem; max-width: 1200px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.2rem; margin: 1.5rem 0 0.75rem; color: var(--accent); }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}

  /* Stats cards */
  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem; margin-bottom: 2rem;
  }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem; text-align: center;
  }}
  .stat-value {{ font-size: 2rem; font-weight: 700; }}
  .stat-label {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-green {{ color: var(--green); }}
  .stat-red {{ color: var(--red); }}
  .stat-yellow {{ color: var(--yellow); }}

  /* Search / filter */
  .controls {{
    display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center;
  }}
  .search-box {{
    flex: 1; min-width: 250px; padding: 0.5rem 0.75rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); font-size: 0.9rem;
  }}
  .search-box:focus {{ outline: none; border-color: var(--accent); }}
  .filter-btn {{
    padding: 0.4rem 0.8rem; background: var(--surface);
    border: 1px solid var(--border); border-radius: 6px;
    color: var(--muted); cursor: pointer; font-size: 0.85rem;
  }}
  .filter-btn:hover, .filter-btn.active {{ color: var(--accent); border-color: var(--accent); }}
  .count-badge {{
    color: var(--muted); font-size: 0.85rem; white-space: nowrap;
  }}

  /* Merge groups */
  .group {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 0.75rem; overflow: hidden;
  }}
  .group-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 1rem; cursor: pointer; user-select: none;
  }}
  .group-header:hover {{ background: rgba(88, 166, 255, 0.05); }}
  .group-title {{ font-weight: 600; }}
  .group-meta {{ color: var(--muted); font-size: 0.85rem; display: flex; gap: 1rem; }}
  .group-body {{ display: none; border-top: 1px solid var(--border); }}
  .group.open .group-body {{ display: block; }}
  .member {{
    display: grid; grid-template-columns: 1fr auto auto auto;
    gap: 0.5rem; padding: 0.5rem 1rem; align-items: center;
    font-size: 0.9rem; border-bottom: 1px solid var(--border);
  }}
  .member:last-child {{ border-bottom: none; }}
  .member.canonical {{ background: var(--canon-bg); }}
  .member-name {{ font-weight: 500; }}
  .member-type {{
    font-size: 0.75rem; padding: 0.15rem 0.5rem;
    background: rgba(88, 166, 255, 0.1); border-radius: 4px;
    color: var(--accent); text-align: center;
  }}
  .member-docs {{ color: var(--muted); font-size: 0.8rem; text-align: right; white-space: nowrap; }}
  .member-badge {{
    font-size: 0.7rem; padding: 0.1rem 0.4rem;
    background: var(--green); color: #000; border-radius: 3px;
    font-weight: 600;
  }}
  .arrow {{ color: var(--muted); transition: transform 0.15s; }}
  .group.open .arrow {{ transform: rotate(90deg); }}
</style>
</head>
<body>

<h1>Entity Normalization Report</h1>
<p class="subtitle">
  <strong>{stats['source_db']}</strong> &rarr; <strong>{stats['output_db']}</strong>
  &nbsp;&middot;&nbsp; {stats['elapsed_seconds']}s
</p>

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">{stats['original_entities']}</div>
    <div class="stat-label">Original Entities</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-green">{stats['final_entities']}</div>
    <div class="stat-label">After Normalization</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-red">{stats['entities_removed']}</div>
    <div class="stat-label">Entities Merged Away</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-yellow">{stats['merge_groups']}</div>
    <div class="stat-label">Merge Groups</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{stats['original_relationships']}</div>
    <div class="stat-label">Original Relationships</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-green">{stats['final_relationships']}</div>
    <div class="stat-label">After Dedup</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{stats['relationships_rewritten']}</div>
    <div class="stat-label">Rels Rewritten</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-red">{stats['relationships_deduped']}</div>
    <div class="stat-label">Rels Deduped</div>
  </div>
</div>

<h2>Merge Groups</h2>

<div class="controls">
  <input type="text" class="search-box" id="search"
         placeholder="Search entity names..." oninput="applyFilters()">
  <button class="filter-btn active" data-size="all" onclick="setFilter(this)">All</button>
  <button class="filter-btn" data-size="3" onclick="setFilter(this)">3+ members</button>
  <button class="filter-btn" data-size="5" onclick="setFilter(this)">5+ members</button>
  <button class="filter-btn" id="expand-all" onclick="toggleAll()">Expand All</button>
  <span class="count-badge" id="count-badge"></span>
</div>

<div id="groups-container"></div>

<script>
const GROUPS = {json.dumps(groups_json)};
let activeMinSize = 0;
let allExpanded = false;

function renderGroups(groups) {{
  const container = document.getElementById('groups-container');
  container.innerHTML = '';
  groups.forEach((g, i) => {{
    const div = document.createElement('div');
    div.className = 'group';
    div.dataset.idx = i;

    const membersHtml = g.members.map(m => {{
      const badge = m.is_canonical ? '<span class="member-badge">KEEP</span>' : '';
      return `<div class="member ${{m.is_canonical ? 'canonical' : ''}}">
        <span class="member-name">${{esc(m.name)}}</span>
        <span class="member-type">${{esc(m.type)}}</span>
        <span class="member-docs">${{m.doc_count}} doc${{m.doc_count !== 1 ? 's' : ''}}</span>
        <span>${{badge}}</span>
      </div>`;
    }}).join('');

    div.innerHTML = `
      <div class="group-header" onclick="this.parentElement.classList.toggle('open')">
        <span class="group-title">${{esc(g.canonical)}}</span>
        <span class="group-meta">
          <span>${{g.member_count}} variants</span>
          <span>key: ${{esc(g.normalized_key)}}</span>
          <span class="arrow">&#9654;</span>
        </span>
      </div>
      <div class="group-body">${{membersHtml}}</div>`;
    container.appendChild(div);
  }});
  document.getElementById('count-badge').textContent = `Showing ${{groups.length}} of ${{GROUPS.length}} groups`;
}}

function esc(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}

function applyFilters() {{
  const q = document.getElementById('search').value.toLowerCase();
  const filtered = GROUPS.filter(g => {{
    if (g.member_count < activeMinSize) return false;
    if (!q) return true;
    return g.members.some(m => m.name.toLowerCase().includes(q)) || g.normalized_key.includes(q);
  }});
  renderGroups(filtered);
}}

function setFilter(btn) {{
  document.querySelectorAll('.filter-btn[data-size]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  activeMinSize = btn.dataset.size === 'all' ? 0 : parseInt(btn.dataset.size);
  applyFilters();
}}

function toggleAll() {{
  allExpanded = !allExpanded;
  document.querySelectorAll('.group').forEach(g => {{
    if (allExpanded) g.classList.add('open'); else g.classList.remove('open');
  }});
  document.getElementById('expand-all').textContent = allExpanded ? 'Collapse All' : 'Expand All';
}}

renderGroups(GROUPS);
</script>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 entity deduplication: deterministic name normalization"
    )
    parser.add_argument(
        "--input", default="BAYT_PROD_entities.db",
        help="Source entity database (default: BAYT_PROD_entities.db)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output entity database (default: <input>_normalized.db)",
    )
    parser.add_argument(
        "--report", default=None,
        help="Output HTML report path (default: <output>_report.html)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print stats and generate report — don't write output DB",
    )
    args = parser.parse_args()

    source_db = args.input
    if not Path(source_db).exists():
        print(f"ERROR: Source database '{source_db}' not found!")
        sys.exit(1)

    stem = Path(source_db).stem
    output_db = args.output or f"{stem}_normalized.db"
    report_path = args.report or f"{Path(output_db).stem}_report.html"

    print("Entity Normalization (Phase 1)")
    print("=" * 60)
    print(f"  Input:  {source_db}")
    print(f"  Output: {output_db}")
    print(f"  Report: {report_path}")
    print()

    t0 = time.time()

    # --- Read source entities -------------------------------------------------
    conn = sqlite3.connect(source_db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM entities")
    entities = [dict(row) for row in c.fetchall()]
    original_entity_count = len(entities)

    c.execute("SELECT COUNT(*) FROM relationships")
    original_rel_count = c.fetchone()[0]

    conn.close()

    print(f"  Source entities:      {original_entity_count:,}")
    print(f"  Source relationships: {original_rel_count:,}")

    # --- Build merge groups ---------------------------------------------------
    merge_groups, name_map = build_merge_groups(entities)

    entities_affected = sum(len(g["members"]) for g in merge_groups)
    entities_removed = entities_affected - len(merge_groups)

    print(f"\n  Merge groups found:  {len(merge_groups):,}")
    print(f"  Entities involved:   {entities_affected:,}")
    print(f"  Will be removed:     {entities_removed:,}")

    # Show some examples
    print(f"\n  Sample merges (up to 15):")
    for mg in sorted(merge_groups, key=lambda g: -len(g["members"]))[:15]:
        names = [ent["name"] for ent in mg["members"]]
        canonical = mg["canonical"]["name"]
        others = [n for n in names if n != canonical]
        print(f"    {canonical}")
        for other in others[:5]:
            print(f"      <- {other}")
        if len(others) > 5:
            print(f"      ... and {len(others) - 5} more")

    # --- Write output DB ------------------------------------------------------
    if not args.dry_run:
        print(f"\n  Writing output database...")
        rewrite_stats = rewrite_database(source_db, output_db, name_map, merge_groups)
        print(f"  Final entities:      {rewrite_stats['final_entity_count']:,}")
        print(f"  Final relationships: {rewrite_stats['final_rel_count']:,}")
        print(f"  Rels rewritten:      {rewrite_stats['relationships_rewritten']:,}")
        print(f"  Rels deduped:        {rewrite_stats['relationships_deduped']:,}")
    else:
        rewrite_stats = {
            "final_entity_count": original_entity_count - entities_removed,
            "final_rel_count": original_rel_count,
            "relationships_rewritten": 0,
            "relationships_deduped": 0,
        }
        print(f"\n  [DRY RUN] — no output DB written")

    elapsed = time.time() - t0

    # --- Generate HTML report -------------------------------------------------
    print(f"\n  Generating report...")
    html = generate_report(
        merge_groups, name_map, source_db, output_db,
        original_entity_count, original_rel_count,
        rewrite_stats, elapsed,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report: {report_path}")

    # --- Summary --------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed:.2f}s")
    print(f"  Entities:      {original_entity_count:,} -> {rewrite_stats['final_entity_count']:,} "
          f"(-{original_entity_count - rewrite_stats['final_entity_count']:,})")
    print(f"  Relationships: {original_rel_count:,} -> {rewrite_stats['final_rel_count']:,}")
    print(f"  Open {report_path} in a browser to inspect merge decisions")


if __name__ == "__main__":
    main()
