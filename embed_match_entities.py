"""
embed_match_entities.py
=======================
Phase 4 entity deduplication: semantic embedding similarity.

Catches duplicates that string matching misses — entities with very
different names but similar descriptions, types, and document context.
For example:
  - "DOE, JOHN A." (description: "Site inspector for ACME") matching
    "John Doe" (description: "Lead inspector at ACME Corp")
  - "EPA" matching "Environmental Protection Agency"
  - Acronyms, alternate names, role-based references

Each entity is embedded as a composite text: "{name} ({type}): {description}".
Cosine similarity between embeddings identifies semantic matches.

Blocking strategy:
  To avoid O(n^2) comparisons, entities are grouped by type, then an
  approximate nearest-neighbor search (brute-force cosine on the
  normalized embedding matrix) finds the top-k neighbors per entity.
  Only pairs above the similarity threshold become candidates.

Usage:
    python embed_match_entities.py                                  # defaults
    python embed_match_entities.py --input normalized_fuzzy.db      # custom
    python embed_match_entities.py --threshold 0.80 --top-k 10      # tuning
    python embed_match_entities.py --dry-run                        # report only

Requirements:
    pip install sentence-transformers numpy
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from normalize_entities import normalize_name, rewrite_database


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.82
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_entity_text(entity: dict) -> str:
    """
    Build composite text for embedding an entity.

    Format: "name (type): description"
    This gives the embedding model context about what the entity is,
    not just its name.
    """
    name = entity.get("name", "")
    etype = entity.get("type", "")
    desc = entity.get("description", "") or ""
    return f"{name} ({etype}): {desc}".strip()


def embed_entities(entities: list, model_name: str = DEFAULT_MODEL, batch_size: int = 64):
    """
    Embed all entities using a sentence transformer model.

    Args:
        entities: list of entity dicts
        model_name: sentence-transformers model name
        batch_size: encoding batch size

    Returns:
        embeddings: numpy array of shape (n_entities, embedding_dim), L2-normalized
    """
    texts = [build_entity_text(ent) for ent in entities]

    print(f"  Loading model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"  Encoding {len(texts):,} entities (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize so dot product = cosine sim
    )

    return embeddings


# ---------------------------------------------------------------------------
# Candidate pair finding via top-k cosine similarity
# ---------------------------------------------------------------------------

def find_candidate_pairs(entities, embeddings, threshold, top_k):
    """
    Find candidate pairs using cosine similarity on embeddings.

    Uses a brute-force approach: compute the similarity matrix and extract
    the top-k neighbors per entity. For entity counts in the thousands
    this is fast enough (matrix multiply on normalized vectors).

    Args:
        entities: list of entity dicts
        embeddings: (n, d) numpy array, L2-normalized
        threshold: minimum cosine similarity
        top_k: number of nearest neighbors to consider per entity

    Returns:
        pairs: list of (entity_a, entity_b, score_dict) sorted by score desc
    """
    n = len(entities)
    if n == 0:
        return []

    # Cosine similarity matrix (since embeddings are L2-normalized, dot = cosine)
    print(f"  Computing similarity matrix ({n:,} x {n:,})...")
    sim_matrix = embeddings @ embeddings.T

    # Zero out self-similarity and below-diagonal to avoid duplicates
    np.fill_diagonal(sim_matrix, 0.0)

    # Build normalized name set to skip pairs Phase 1 already caught
    norm_keys = {}
    for ent in entities:
        norm_keys[ent["name"]] = normalize_name(ent["name"])

    pairs = []
    seen = set()

    for i in range(n):
        # Get top-k indices for entity i
        if n <= top_k:
            top_indices = np.argsort(sim_matrix[i])[::-1]
        else:
            # Partial sort: only need top_k
            top_indices = np.argpartition(sim_matrix[i], -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sim_matrix[i][top_indices])[::-1]]

        for j in top_indices:
            if j <= i:
                continue  # only upper triangle
            score = float(sim_matrix[i, j])
            if score < threshold:
                continue

            name_a = entities[i]["name"]
            name_b = entities[j]["name"]

            # Skip if Phase 1 normalization would already catch this pair
            if norm_keys[name_a] == norm_keys[name_b]:
                continue

            pair_key = (min(name_a, name_b), max(name_a, name_b))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            pairs.append((entities[i], entities[j], {
                "cosine_sim": round(score, 4),
                "text_a": build_entity_text(entities[i]),
                "text_b": build_entity_text(entities[j]),
            }))

    pairs.sort(key=lambda x: -x[2]["cosine_sim"])
    return pairs


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
    Cluster embedding-match pairs into merge groups using union-find.

    Returns:
        merge_groups: list of merge group dicts
        name_map: dict mapping original_name -> canonical_name
    """
    if not pairs:
        return [], {ent["name"]: ent["name"] for ent in entities}

    uf = UnionFind()
    for ent_a, ent_b, _ in pairs:
        uf.union(ent_a["name"], ent_b["name"])

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
    rewrite_stats, threshold, model_name, elapsed,
):
    """Generate HTML report for embedding match inspection."""

    pairs_json = []
    for ent_a, ent_b, scores in pairs[:500]:
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
            "text_a": scores["text_a"],
            "name_b": ent_b["name"],
            "type_b": ent_b["type"],
            "desc_b": ent_b["description"] or "",
            "docs_b": docs_b,
            "text_b": scores["text_b"],
            "cosine": scores["cosine_sim"],
            "canonical": canonical,
        })

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
        "model": model_name,
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
<title>Embedding Match Report (Phase 4)</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --orange: #d18616; --canon-bg: #1a2332; --purple: #bc8cff;
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
  .stat-purple {{ color: var(--purple); }}

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

  /* Pair cards (not a table — descriptions are long) */
  .pair-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 0.75rem; padding: 1rem;
    display: grid; grid-template-columns: 1fr auto 1fr; gap: 1rem; align-items: start;
  }}
  .pair-entity {{ font-size: 0.85rem; }}
  .pair-entity .name {{ font-weight: 600; font-size: 0.95rem; }}
  .pair-entity .desc {{ color: var(--muted); margin-top: 0.25rem; font-style: italic; }}
  .pair-entity .meta {{ color: var(--muted); font-size: 0.8rem; margin-top: 0.2rem; }}
  .pair-score {{
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-width: 80px;
  }}
  .pair-score .value {{ font-size: 1.4rem; font-weight: 700; }}
  .pair-score .label {{ font-size: 0.7rem; color: var(--muted); text-transform: uppercase; }}
  .pair-score .canonical {{ font-size: 0.7rem; color: var(--green); margin-top: 0.3rem; }}
  .score-high {{ color: var(--green); }}
  .score-mid {{ color: var(--yellow); }}
  .score-low {{ color: var(--orange); }}
  .type-badge {{
    font-size: 0.7rem; padding: 0.1rem 0.4rem;
    background: rgba(88, 166, 255, 0.1); border-radius: 3px;
    color: var(--accent); margin-left: 0.3rem;
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
    display: grid; grid-template-columns: 1fr 1fr auto auto auto;
    gap: 0.5rem; padding: 0.5rem 1rem; align-items: center;
    font-size: 0.85rem; border-bottom: 1px solid var(--border);
  }}
  .member:last-child {{ border-bottom: none; }}
  .member.canonical {{ background: var(--canon-bg); }}
  .member-name {{ font-weight: 500; }}
  .member-desc {{ color: var(--muted); font-style: italic; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .member-type {{
    font-size: 0.7rem; padding: 0.1rem 0.4rem;
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

<h1>Embedding Match Report (Phase 4)</h1>
<p class="subtitle">
  <strong>{stats['source_db']}</strong> &rarr; <strong>{stats['output_db']}</strong>
  &nbsp;&middot;&nbsp; model: {stats['model']}
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
    <div class="stat-label">After Embed Merge</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-red">{stats['entities_removed']}</div>
    <div class="stat-label">Merged Away</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-purple">{stats['candidate_pairs']}</div>
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
  <div class="tab active" onclick="switchTab('pairs')">Candidate Pairs ({len(pairs)})</div>
  <div class="tab" onclick="switchTab('groups')">Merge Groups ({len(merge_groups)})</div>
</div>

<div id="tab-pairs" class="tab-content active">
  <div class="controls">
    <input type="text" class="search-box" id="pair-search"
           placeholder="Search by name or description..." oninput="filterPairs()">
    <span class="count-badge" id="pair-count"></span>
  </div>
  <div id="pairs-container"></div>
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
let allExpanded = false;

function esc(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}
function scoreClass(v) {{ return v >= 0.90 ? 'score-high' : v >= 0.85 ? 'score-mid' : 'score-low'; }}

function renderPairs(pairs) {{
  const container = document.getElementById('pairs-container');
  container.innerHTML = pairs.map(p => `
    <div class="pair-card">
      <div class="pair-entity">
        <div class="name">${{esc(p.name_a)}}<span class="type-badge">${{p.type_a}}</span></div>
        <div class="desc">${{esc(p.desc_a || '(no description)')}}</div>
        <div class="meta">${{p.docs_a}} doc${{p.docs_a !== 1 ? 's' : ''}}</div>
      </div>
      <div class="pair-score">
        <div class="value ${{scoreClass(p.cosine)}}">${{p.cosine.toFixed(3)}}</div>
        <div class="label">cosine</div>
        <div class="canonical">keep: ${{esc(p.canonical)}}</div>
      </div>
      <div class="pair-entity" style="text-align:right">
        <div class="name"><span class="type-badge">${{p.type_b}}</span>${{esc(p.name_b)}}</div>
        <div class="desc">${{esc(p.desc_b || '(no description)')}}</div>
        <div class="meta">${{p.docs_b}} doc${{p.docs_b !== 1 ? 's' : ''}}</div>
      </div>
    </div>
  `).join('');
  document.getElementById('pair-count').textContent = `${{pairs.length}} of ${{PAIRS.length}} pairs`;
}}

function filterPairs() {{
  const q = document.getElementById('pair-search').value.toLowerCase();
  const filtered = q ? PAIRS.filter(p =>
    p.name_a.toLowerCase().includes(q) || p.name_b.toLowerCase().includes(q) ||
    p.desc_a.toLowerCase().includes(q) || p.desc_b.toLowerCase().includes(q)
  ) : PAIRS;
  renderPairs(filtered);
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
        <span class="member-desc">${{esc(m.description || '(none)')}}</span>
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
    g.members.some(m => m.name.toLowerCase().includes(q) || m.description.toLowerCase().includes(q))
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
        description="Phase 4 entity deduplication: semantic embedding similarity"
    )
    parser.add_argument(
        "--input", default="BAYT_PROD_entities_normalized_fuzzy.db",
        help="Source entity database (default: BAYT_PROD_entities_normalized_fuzzy.db)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output entity database (default: <input_stem>_embed.db)",
    )
    parser.add_argument(
        "--report", default=None,
        help="Output HTML report path (default: <output_stem>_report.html)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Minimum cosine similarity to merge (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Nearest neighbors per entity to consider (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Sentence-transformers model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only find pairs and generate report -- don't write output DB",
    )
    args = parser.parse_args()

    source_db = args.input
    if not Path(source_db).exists():
        print(f"ERROR: Source database '{source_db}' not found!")
        sys.exit(1)

    stem = Path(source_db).stem
    output_db = args.output or f"{stem}_embed.db"
    report_path = args.report or f"{Path(output_db).stem}_report.html"

    print("Entity Embedding Match (Phase 4)")
    print("=" * 60)
    print(f"  Input:     {source_db}")
    print(f"  Output:    {output_db}")
    print(f"  Report:    {report_path}")
    print(f"  Model:     {args.model}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Top-K:     {args.top_k}")
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

    # --- Embed entities -------------------------------------------------------
    embeddings = embed_entities(entities, model_name=args.model)
    embed_time = time.time() - t0
    print(f"  Embedding time: {embed_time:.1f}s")

    # --- Find candidate pairs -------------------------------------------------
    print(f"\n  Finding candidate pairs (threshold={args.threshold}, top_k={args.top_k})...")
    pairs = find_candidate_pairs(entities, embeddings, args.threshold, args.top_k)
    print(f"  Candidate pairs: {len(pairs):,}")

    if pairs:
        print(f"\n  Top matches:")
        for ent_a, ent_b, scores in pairs[:15]:
            print(f"    {scores['cosine_sim']:.3f}  {ent_a['name']!r:30s} <-> {ent_b['name']!r}")
            if ent_a["description"] or ent_b["description"]:
                desc_a = (ent_a["description"] or "")[:50]
                desc_b = (ent_b["description"] or "")[:50]
                print(f"           {desc_a!r:52s}     {desc_b!r}")

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
        print(f"\n  [DRY RUN] -- no output DB written")

    elapsed = time.time() - t0

    # --- Generate HTML report -------------------------------------------------
    print(f"\n  Generating report...")
    html = generate_report(
        pairs, merge_groups, name_map,
        source_db, output_db,
        original_entity_count, original_rel_count,
        rewrite_stats, args.threshold, args.model, elapsed,
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
    print(f"  Open {report_path} in a browser to inspect embedding match decisions")


if __name__ == "__main__":
    main()
