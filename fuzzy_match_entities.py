"""
fuzzy_match_entities.py
=======================
Phase 2 entity deduplication: string similarity matching.

Reads a DORA entity database (ideally the output of Phase 1 normalization),
finds fuzzy-match candidate pairs using token-blocking + multi-metric scoring,
clusters them via union-find, and writes a new database with merged entities
and updated relationships.

Produces an HTML inspection report showing candidate pairs with scores
so you can tune thresholds before trusting merges.

Metrics used:
  - Jaro-Winkler similarity  (good for typos: "Jon" vs "John")
  - Token-set ratio           (good for reordering / partial overlap)
  - Combined score = max(jw, token_set / 100)

Blocking strategy:
  Pairs must share at least one token (word) in the normalized name.
  This avoids O(n^2) comparisons while catching all interesting pairs.

Usage:
    python fuzzy_match_entities.py                              # defaults
    python fuzzy_match_entities.py --input normalized.db        # custom input
    python fuzzy_match_entities.py --threshold 0.85             # stricter
    python fuzzy_match_entities.py --dry-run                    # report only
    python fuzzy_match_entities.py --threshold 0.80 --input x.db --output y.db

Requirements:
    pip install rapidfuzz
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

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from normalize_entities import normalize_name, rewrite_database


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_pair(name_a: str, name_b: str) -> dict:
    """
    Compute multiple similarity metrics between two entity names.

    Returns dict with individual scores and a combined score.
    All scores are in [0, 1].
    """
    # Jaro-Winkler on the full normalized name
    jw = JaroWinkler.normalized_similarity(name_a, name_b)

    # Token-set ratio: handles reordering and subset matching
    # e.g. "John A. Smith" vs "Smith, John" -> high score
    tsr = fuzz.token_set_ratio(name_a, name_b) / 100.0

    # Partial ratio: best substring alignment
    # e.g. "EPA" vs "US EPA" -> high partial
    partial = fuzz.partial_ratio(name_a, name_b) / 100.0

    combined = max(jw, tsr)

    return {
        "jaro_winkler": round(jw, 4),
        "token_set_ratio": round(tsr, 4),
        "partial_ratio": round(partial, 4),
        "combined": round(combined, 4),
    }


# ---------------------------------------------------------------------------
# Blocking: token-based inverted index
# ---------------------------------------------------------------------------

# Tokens too common to be useful as blocking keys
STOP_TOKENS = {
    "the", "of", "and", "for", "in", "on", "at", "to", "a", "an",
    "is", "was", "are", "by", "from", "with", "as", "or", "not",
    "no", "its", "it", "be", "has", "had", "have",
    # common entity type words that appear in many names
    "project", "site", "phase", "section", "report", "letter",
    "inspection", "review", "meeting", "issue", "item", "area",
}

MIN_TOKEN_LEN = 2  # skip single-character tokens as blocking keys


def tokenize_for_blocking(name: str) -> set:
    """Extract blocking tokens from a normalized name."""
    tokens = set(name.split())
    return {t for t in tokens if len(t) >= MIN_TOKEN_LEN and t not in STOP_TOKENS}


def build_candidate_pairs(entities, threshold):
    """
    Build candidate pairs using token-blocking, then score them.

    Args:
        entities: list of entity dicts (with 'name' field)
        threshold: minimum combined score to keep a pair

    Returns:
        pairs: list of (entity_a, entity_b, scores_dict)
    """
    # Normalize all names and build token index
    normed = {}  # entity name -> normalized name
    token_index = defaultdict(set)  # token -> set of entity names

    for ent in entities:
        name = ent["name"]
        nname = normalize_name(name)
        normed[name] = nname
        for token in tokenize_for_blocking(nname):
            token_index[token].add(name)

    # Generate candidate pairs via shared tokens
    candidate_set = set()
    for token, names in token_index.items():
        names_list = sorted(names)
        for i in range(len(names_list)):
            for j in range(i + 1, len(names_list)):
                pair = (names_list[i], names_list[j])
                candidate_set.add(pair)

    # Score candidates
    entity_lookup = {ent["name"]: ent for ent in entities}
    pairs = []

    for name_a, name_b in candidate_set:
        nname_a = normed[name_a]
        nname_b = normed[name_b]

        # Skip if normalized names are identical (Phase 1 should have caught these)
        if nname_a == nname_b:
            continue

        scores = score_pair(nname_a, nname_b)
        if scores["combined"] >= threshold:
            pairs.append((entity_lookup[name_a], entity_lookup[name_b], scores))

    # Sort by combined score descending
    pairs.sort(key=lambda x: -x[2]["combined"])

    return pairs


# ---------------------------------------------------------------------------
# Union-Find (same pattern as greedy_merge.py)
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
            self.parent[x] = self.parent[self.parent[x]]
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

    def components(self):
        groups = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return dict(groups)


def cluster_pairs(pairs, entities):
    """
    Cluster fuzzy-match pairs into merge groups using union-find.

    Args:
        pairs: list of (entity_a, entity_b, scores) from build_candidate_pairs
        entities: full entity list (for doc-count based canonical selection)

    Returns:
        merge_groups: list of merge group dicts (same format as Phase 1)
        name_map: dict mapping original_name -> canonical_name
    """
    if not pairs:
        return [], {ent["name"]: ent["name"] for ent in entities}

    uf = UnionFind()
    for ent_a, ent_b, _ in pairs:
        uf.union(ent_a["name"], ent_b["name"])

    # Ensure all entities are in the UF (singletons map to themselves)
    for ent in entities:
        uf.find(ent["name"])

    components = uf.components()
    entity_lookup = {ent["name"]: ent for ent in entities}

    name_map = {}
    merge_groups = []

    for root, names in components.items():
        group_entities = [entity_lookup[n] for n in names if n in entity_lookup]
        if not group_entities:
            continue

        # Pick canonical: most doc appearances, then longest name
        def sort_key(e):
            try:
                doc_count = len(json.loads(e["documents_appeared"])) if e["documents_appeared"] else 0
            except (json.JSONDecodeError, TypeError):
                doc_count = 0
            return (-doc_count, -len(e["name"]))

        group_entities.sort(key=sort_key)
        canonical = group_entities[0]

        for ent in group_entities:
            name_map[ent["name"]] = canonical["name"]

        if len(group_entities) > 1:
            merge_groups.append({
                "canonical": canonical,
                "members": group_entities,
                "normalized_key": normalize_name(canonical["name"]),
            })

    return merge_groups, name_map


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_report(
    pairs, merge_groups, name_map,
    source_db, output_db,
    original_entity_count, original_rel_count,
    rewrite_stats, threshold, elapsed,
):
    """Generate HTML report for fuzzy match inspection."""

    # Prepare pairs data for the report
    pairs_json = []
    for ent_a, ent_b, scores in pairs[:500]:  # cap at 500 for report size
        try:
            docs_a = len(json.loads(ent_a["documents_appeared"])) if ent_a["documents_appeared"] else 0
        except (json.JSONDecodeError, TypeError):
            docs_a = 0
        try:
            docs_b = len(json.loads(ent_b["documents_appeared"])) if ent_b["documents_appeared"] else 0
        except (json.JSONDecodeError, TypeError):
            docs_b = 0

        canonical = name_map.get(ent_a["name"], ent_a["name"])

        pairs_json.append({
            "name_a": ent_a["name"],
            "type_a": ent_a["type"],
            "desc_a": ent_a["description"] or "",
            "docs_a": docs_a,
            "name_b": ent_b["name"],
            "type_b": ent_b["type"],
            "desc_b": ent_b["description"] or "",
            "docs_b": docs_b,
            "jw": scores["jaro_winkler"],
            "tsr": scores["token_set_ratio"],
            "partial": scores["partial_ratio"],
            "combined": scores["combined"],
            "canonical": canonical,
        })

    # Prepare merge group data
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
            "member_count": len(mg["members"]),
            "members": members,
        })

    stats = {
        "source_db": source_db,
        "output_db": output_db,
        "threshold": threshold,
        "original_entities": original_entity_count,
        "final_entities": rewrite_stats["final_entity_count"],
        "entities_removed": original_entity_count - rewrite_stats["final_entity_count"],
        "original_relationships": original_rel_count,
        "final_relationships": rewrite_stats["final_rel_count"],
        "relationships_rewritten": rewrite_stats["relationships_rewritten"],
        "relationships_deduped": rewrite_stats["relationships_deduped"],
        "candidate_pairs": len(pairs),
        "merge_groups": len(merge_groups),
        "elapsed_seconds": round(elapsed, 2),
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fuzzy Match Report</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --orange: #d18616; --canon-bg: #1a2332;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.5; padding: 2rem; max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.2rem; margin: 1.5rem 0 0.75rem; color: var(--accent); }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}

  .stats-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.75rem; margin-bottom: 2rem;
  }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.75rem; text-align: center;
  }}
  .stat-value {{ font-size: 1.6rem; font-weight: 700; }}
  .stat-label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-green {{ color: var(--green); }}
  .stat-red {{ color: var(--red); }}
  .stat-yellow {{ color: var(--yellow); }}

  .tab-bar {{
    display: flex; gap: 0; margin-bottom: 1rem; border-bottom: 1px solid var(--border);
  }}
  .tab {{
    padding: 0.5rem 1.2rem; cursor: pointer; color: var(--muted);
    border-bottom: 2px solid transparent; font-size: 0.9rem;
  }}
  .tab:hover {{ color: var(--text); }}
  .tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

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
  .count-badge {{ color: var(--muted); font-size: 0.85rem; white-space: nowrap; }}

  /* Pairs table */
  .pairs-table {{
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
  }}
  .pairs-table th {{
    text-align: left; padding: 0.5rem 0.6rem; border-bottom: 2px solid var(--border);
    color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
    position: sticky; top: 0; background: var(--bg); cursor: pointer;
  }}
  .pairs-table th:hover {{ color: var(--accent); }}
  .pairs-table td {{
    padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border);
  }}
  .pairs-table tr:hover {{ background: rgba(88, 166, 255, 0.03); }}
  .score-bar {{
    display: inline-block; height: 6px; border-radius: 3px;
    min-width: 4px; vertical-align: middle; margin-right: 4px;
  }}
  .score-high {{ background: var(--green); }}
  .score-mid {{ background: var(--yellow); }}
  .score-low {{ background: var(--orange); }}
  .type-badge {{
    font-size: 0.7rem; padding: 0.1rem 0.4rem;
    background: rgba(88, 166, 255, 0.1); border-radius: 3px;
    color: var(--accent);
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
    background: var(--green); color: #000; border-radius: 3px; font-weight: 600;
  }}
  .arrow {{ color: var(--muted); transition: transform 0.15s; }}
  .group.open .arrow {{ transform: rotate(90deg); }}
</style>
</head>
<body>

<h1>Fuzzy Match Report (Phase 2)</h1>
<p class="subtitle">
  <strong>{stats['source_db']}</strong> &rarr; <strong>{stats['output_db']}</strong>
  &nbsp;&middot;&nbsp; threshold: {stats['threshold']}
  &nbsp;&middot;&nbsp; {stats['elapsed_seconds']}s
</p>

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">{stats['original_entities']}</div>
    <div class="stat-label">Input Entities</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-green">{stats['final_entities']}</div>
    <div class="stat-label">After Fuzzy Merge</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-red">{stats['entities_removed']}</div>
    <div class="stat-label">Merged Away</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-yellow">{stats['candidate_pairs']}</div>
    <div class="stat-label">Candidate Pairs</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-yellow">{stats['merge_groups']}</div>
    <div class="stat-label">Merge Groups</div>
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

<div class="tab-bar">
  <div class="tab active" onclick="switchTab('pairs')">Candidate Pairs</div>
  <div class="tab" onclick="switchTab('groups')">Merge Groups ({len(merge_groups)})</div>
</div>

<div id="tab-pairs" class="tab-content active">
  <div class="controls">
    <input type="text" class="search-box" id="pair-search"
           placeholder="Search pairs..." oninput="filterPairs()">
    <button class="filter-btn" id="sort-combined" onclick="sortPairs('combined')">Sort: Combined</button>
    <button class="filter-btn" onclick="sortPairs('jw')">Sort: JW</button>
    <button class="filter-btn" onclick="sortPairs('tsr')">Sort: TokenSet</button>
    <span class="count-badge" id="pair-count"></span>
  </div>
  <table class="pairs-table">
    <thead>
      <tr>
        <th>Entity A</th>
        <th>Entity B</th>
        <th onclick="sortPairs('combined')">Combined</th>
        <th onclick="sortPairs('jw')">JW</th>
        <th onclick="sortPairs('tsr')">TokenSet</th>
        <th>Canonical</th>
      </tr>
    </thead>
    <tbody id="pairs-body"></tbody>
  </table>
</div>

<div id="tab-groups" class="tab-content">
  <div class="controls">
    <input type="text" class="search-box" id="group-search"
           placeholder="Search groups..." oninput="filterGroups()">
    <button class="filter-btn" id="expand-all" onclick="toggleAll()">Expand All</button>
    <span class="count-badge" id="group-count"></span>
  </div>
  <div id="groups-container"></div>
</div>

<script>
const PAIRS = {json.dumps(pairs_json)};
const GROUPS = {json.dumps(groups_json)};
let currentSort = 'combined';
let allExpanded = false;

function esc(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}

function scoreClass(v) {{ return v >= 0.92 ? 'score-high' : v >= 0.85 ? 'score-mid' : 'score-low'; }}
function scoreBar(v) {{
  const w = Math.round(v * 60);
  return `<span class="score-bar ${{scoreClass(v)}}" style="width:${{w}}px"></span>${{v.toFixed(3)}}`;
}}

function renderPairs(pairs) {{
  const tbody = document.getElementById('pairs-body');
  tbody.innerHTML = pairs.map(p => `<tr>
    <td>${{esc(p.name_a)}} <span class="type-badge">${{p.type_a}}</span></td>
    <td>${{esc(p.name_b)}} <span class="type-badge">${{p.type_b}}</span></td>
    <td>${{scoreBar(p.combined)}}</td>
    <td>${{scoreBar(p.jw)}}</td>
    <td>${{scoreBar(p.tsr)}}</td>
    <td style="color:var(--green);font-size:0.8rem">${{esc(p.canonical)}}</td>
  </tr>`).join('');
  document.getElementById('pair-count').textContent = `${{pairs.length}} of ${{PAIRS.length}} pairs`;
}}

function filterPairs() {{
  const q = document.getElementById('pair-search').value.toLowerCase();
  const filtered = q ? PAIRS.filter(p =>
    p.name_a.toLowerCase().includes(q) || p.name_b.toLowerCase().includes(q)
  ) : [...PAIRS];
  filtered.sort((a, b) => b[currentSort] - a[currentSort]);
  renderPairs(filtered);
}}

function sortPairs(field) {{
  currentSort = field;
  filterPairs();
}}

function renderGroups(groups) {{
  const container = document.getElementById('groups-container');
  container.innerHTML = '';
  groups.forEach(g => {{
    const div = document.createElement('div');
    div.className = 'group';
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
          <span class="arrow">&#9654;</span>
        </span>
      </div>
      <div class="group-body">${{membersHtml}}</div>`;
    container.appendChild(div);
  }});
  document.getElementById('group-count').textContent = `${{groups.length}} groups`;
}}

function filterGroups() {{
  const q = document.getElementById('group-search').value.toLowerCase();
  const filtered = q ? GROUPS.filter(g =>
    g.members.some(m => m.name.toLowerCase().includes(q))
  ) : GROUPS;
  renderGroups(filtered);
}}

function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab-content#tab-${{name}}`).classList.add('active');
  event.target.classList.add('active');
}}

function toggleAll() {{
  allExpanded = !allExpanded;
  document.querySelectorAll('.group').forEach(g => {{
    if (allExpanded) g.classList.add('open'); else g.classList.remove('open');
  }});
  document.getElementById('expand-all').textContent = allExpanded ? 'Collapse All' : 'Expand All';
}}

renderPairs(PAIRS);
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
        description="Phase 2 entity deduplication: string similarity matching"
    )
    parser.add_argument(
        "--input", default="BAYT_PROD_entities_normalized.db",
        help="Source entity database (default: BAYT_PROD_entities_normalized.db)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output entity database (default: <input_stem>_fuzzy.db)",
    )
    parser.add_argument(
        "--report", default=None,
        help="Output HTML report path (default: <output_stem>_report.html)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.88,
        help="Minimum combined similarity score to merge (default: 0.88)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only find pairs and generate report — don't write output DB",
    )
    args = parser.parse_args()

    source_db = args.input
    if not Path(source_db).exists():
        print(f"ERROR: Source database '{source_db}' not found!")
        sys.exit(1)

    stem = Path(source_db).stem
    output_db = args.output or f"{stem}_fuzzy.db"
    report_path = args.report or f"{Path(output_db).stem}_report.html"
    threshold = args.threshold

    print("Entity Fuzzy Matching (Phase 2)")
    print("=" * 60)
    print(f"  Input:     {source_db}")
    print(f"  Output:    {output_db}")
    print(f"  Report:    {report_path}")
    print(f"  Threshold: {threshold}")
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

    # --- Find candidate pairs -------------------------------------------------
    print(f"\n  Building candidate pairs (token-blocking)...")
    pairs = build_candidate_pairs(entities, threshold)
    print(f"  Candidate pairs above {threshold}: {len(pairs):,}")

    # Show top pairs
    if pairs:
        print(f"\n  Top matches:")
        for ent_a, ent_b, scores in pairs[:20]:
            print(f"    {scores['combined']:.3f}  {ent_a['name']!r:30s} <-> {ent_b['name']!r}")

    # --- Cluster into merge groups --------------------------------------------
    print(f"\n  Clustering pairs...")
    merge_groups, name_map = cluster_pairs(pairs, entities)

    entities_affected = sum(len(g["members"]) for g in merge_groups)
    entities_removed = entities_affected - len(merge_groups)

    print(f"  Merge groups: {len(merge_groups):,}")
    print(f"  Entities involved: {entities_affected:,}")
    print(f"  Will be removed: {entities_removed:,}")

    if merge_groups:
        print(f"\n  Sample merges:")
        for mg in sorted(merge_groups, key=lambda g: -len(g["members"]))[:10]:
            canonical = mg["canonical"]["name"]
            others = [ent["name"] for ent in mg["members"] if ent["name"] != canonical]
            print(f"    {canonical}")
            for other in others[:5]:
                print(f"      <- {other}")

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
        pairs, merge_groups, name_map,
        source_db, output_db,
        original_entity_count, original_rel_count,
        rewrite_stats, threshold, elapsed,
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
    print(f"  Open {report_path} in a browser to inspect fuzzy match decisions")


if __name__ == "__main__":
    main()
