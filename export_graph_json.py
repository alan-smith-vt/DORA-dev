"""
export_graph_json.py - Export knowledge graph to JSON for 3D viewer.

Reads entity and chunk databases and produces a graph_data.json file
that the 3D HTML viewer loads.

Supports both old and new database schemas:
  OLD: entities have first_seen_chunk / chunks_appeared
       chunks table has no filename column
       relationships have chunk_id / chapter
       no documents table
  NEW: entities have first_seen_document / documents_appeared
       chunks table has filename column
       relationships have document_id
       documents table exists

Usage:
    python export_graph_json.py

Produces:
    - graph_data.json (loaded by correspondence_graph_3d.html)
"""

import json
import sqlite3
from pathlib import Path
import re
from typing import Dict, Optional

# ============================================================================
# CONFIGURATION — change these for your project
# ============================================================================

# ENTITY_DB = "C:/Users/agsmith/Documents/_ProjectsLocal/_LLM_Research/llm/DEAR_LLM/DEAR_entities.db"
# CHUNK_DB = "C:/Users/agsmith/Documents/_ProjectsLocal/_LLM_Research/llm/DEAR_LLM/DEAR_chunks.db"
# OUTPUT_JSON = "data/DEAR_graph_data.json"
# DOC_BASE_PATH = ""                  # Prefix for file paths (e.g. "C:/Projects/Docs/")

# ENTITY_DB = "C:/Users/agsmith/Documents/_ProjectsLocal/_LLM_Research/llm/mother_of_learning_entities.db"
# CHUNK_DB = "C:/Users/agsmith/Documents/_ProjectsLocal/_LLM_Research/llm/mother_of_learning_chunks.db"
# OUTPUT_JSON = "data/MOL_graph_data.json"
# DOC_BASE_PATH = ""                  # Prefix for file paths (e.g. "C:/Projects/Docs/")

# ENTITY_DB = "C:/Users/agsmith/Documents/_ProjectsLocal/_LLM_Research/../251331_OBAM/OBAM_entities.db"
# CHUNK_DB = "C:/Users/agsmith/Documents/_ProjectsLocal/_LLM_Research/../251331_OBAM/OBAM_chunks.db"
# OUTPUT_JSON = "data/OBAM_graph_data.json"
# DOC_BASE_PATH = "I:/CHI/projects/2025/251331.00-OBAM/Working/GraphData/"                  # Prefix for file paths (e.g. "C:/Projects/Docs/")


#ENTITY_DB = "C:/Users/agsmith/Documents/_ProjectsLocal/_LLM_Research/Page364_entities.db"
#CHUNK_DB = "C:/Users/agsmith/Documents/_ProjectsLocal/_LLM_Research/Page364_chunks.db"
#OUTPUT_JSON = "data/Page364_graph_data.json"
#DOC_BASE_PATH = ""                  # Prefix for file paths (e.g. "C:/Projects/Docs/")

ENTITY_DB = "BAYT_PROD_entities.db"
CHUNK_DB = "BAYT_PROD_chunks.db"
OUTPUT_JSON = "BAYT_PROD.json"
DOC_BASE_PATH = ""
                  # Prefix for file paths (e.g. "C:/Projects/Docs/")

# ============================================================================
# Schema Detection
# ============================================================================

def get_table_columns(cursor, table_name: str) -> set:
    """Get column names for a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def get_tables(cursor) -> set:
    """Get all table names in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cursor.fetchall()}


def detect_schema(cursor) -> dict:
    """Detect which schema variant we're dealing with."""
    tables = get_tables(cursor)
    entity_cols = get_table_columns(cursor, 'entities') if 'entities' in tables else set()
    rel_cols = get_table_columns(cursor, 'relationships') if 'relationships' in tables else set()

    schema = {
        'has_documents_table': 'documents' in tables,
        'has_chunks_table': 'chunks' in tables,
        # Entity fields
        'entity_docs_field': 'documents_appeared' if 'documents_appeared' in entity_cols else
                             'chunks_appeared' if 'chunks_appeared' in entity_cols else None,
        'entity_first_seen_field': 'first_seen_document' if 'first_seen_document' in entity_cols else
                                   'first_seen_chunk' if 'first_seen_chunk' in entity_cols else None,
        # Relationship fields
        'rel_doc_field': 'document_id' if 'document_id' in rel_cols else
                         'chunk_id' if 'chunk_id' in rel_cols else None,
    }

    return schema


# ============================================================================
# Chunk-to-file mapping (for new schema with filename in chunks table)
# ============================================================================

def build_chunk_to_file_map(chunk_db_path: str) -> Dict[str, Dict[str, str]]:
    """Build mapping from chunk_id to source filename and page number.
    Only works if chunks table has a 'filename' column."""
    if not chunk_db_path or not Path(chunk_db_path).exists():
        return {}

    conn = sqlite3.connect(chunk_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if filename column exists
    cols = get_table_columns(cursor, 'chunks')
    if 'filename' not in cols:
        print("  (chunks table has no 'filename' column — skipping file mapping)")
        conn.close()
        return {}

    cursor.execute('SELECT chunk_id, filename FROM chunks ORDER BY chunk_id')

    chunk_map = {}
    for row in cursor.fetchall():
        chunk_id = str(row['chunk_id'])
        filename = row['filename']
        page_match = re.search(r'(\.\w+)_(\d+)$', filename)
        if page_match:
            clean_name = filename[:page_match.start()] + page_match.group(1)
            page_num = str(int(page_match.group(2)) + 1)
        else:
            clean_name = filename
            page_num = "1"

        chunk_map[chunk_id] = {
            'filename': clean_name,
            'page': page_num
        }

    conn.close()
    return chunk_map


# ============================================================================
# Chunk-to-chapter mapping (for old schema without filename)
# ============================================================================

def build_chunk_to_chapter_map(cursor_or_path) -> Dict[str, Dict[str, str]]:
    """Build mapping from chunk_id to chapter number (old schema fallback)."""
    # Accept either a cursor (same DB) or a path (separate DB)
    if isinstance(cursor_or_path, str):
        if not Path(cursor_or_path).exists():
            return {}
        conn = sqlite3.connect(cursor_or_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        own_conn = True
    else:
        cursor = cursor_or_path
        own_conn = False

    tables = get_tables(cursor)
    if 'chunks' not in tables:
        if own_conn:
            conn.close()
        return {}

    cols = get_table_columns(cursor, 'chunks')
    chunk_map = {}

    if 'chapter' in cols:
        cursor.execute('SELECT chunk_id, chapter, chunk_index FROM chunks ORDER BY chunk_id')
        for row in cursor.fetchall():
            chunk_id = str(row['chunk_id'])
            chapter = str(row['chapter']) if row['chapter'] is not None else '?'
            chunk_map[chunk_id] = {
                'filename': f"Chapter {chapter}",
                'page': str(row['chunk_index'] + 1) if 'chunk_index' in cols and row['chunk_index'] is not None else '?'
            }
    else:
        cursor.execute('SELECT chunk_id FROM chunks ORDER BY chunk_id')
        for row in cursor.fetchall():
            chunk_map[str(row['chunk_id'])] = {'filename': f"Chunk {row['chunk_id']}", 'page': '?'}

    if own_conn:
        conn.close()
    return chunk_map


# ============================================================================
# Search text builder
# ============================================================================

def build_search_text(cursor, entity_name: str, entity_type: str,
                      description: str, schema: dict) -> str:
    """Build comprehensive search text for an entity."""
    parts = [entity_name, entity_type, description]

    # Try to get document metadata (new schema only)
    if schema['has_documents_table'] and schema['entity_docs_field'] == 'documents_appeared':
        try:
            cursor.execute('''
                SELECT DISTINCT d.subject, d.sender, d.recipient, d.reference
                FROM documents d
                WHERE d.document_id IN (
                    SELECT json_each.value 
                    FROM entities e, json_each(e.documents_appeared)
                    WHERE e.name = ?
                )
            ''', (entity_name,))

            for row in cursor.fetchall():
                for field in ['subject', 'sender', 'recipient', 'reference']:
                    try:
                        val = row[field]
                        if val:
                            parts.append(val)
                    except (IndexError, KeyError):
                        pass
        except Exception:
            pass  # documents table might have different columns

    # Get relationship types and descriptions (works with both schemas)
    try:
        cursor.execute('''
            SELECT DISTINCT type, description
            FROM relationships
            WHERE source = ? OR target = ?
        ''', (entity_name, entity_name))

        for row in cursor.fetchall():
            if row['type']:
                parts.append(row['type'])
            if row['description']:
                parts.append(row['description'])
    except Exception:
        pass

    return ' '.join(filter(None, parts)).lower()


# ============================================================================
# Safe field accessor
# ============================================================================

def safe_get(row, field, default=None):
    """Safely get a field from a sqlite3.Row, returning default if missing."""
    try:
        val = row[field]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default


# ============================================================================
# Main Export
# ============================================================================

def export_graph():
    print("Knowledge Graph JSON Export")
    print("=" * 60)

    if not Path(ENTITY_DB).exists():
        print(f"ERROR: Entity database '{ENTITY_DB}' not found!")
        return 1

    # Connect to entity DB
    conn = sqlite3.connect(ENTITY_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Detect schema
    schema = detect_schema(cursor)
    print(f"Schema detected:")
    print(f"  Documents table: {'yes' if schema['has_documents_table'] else 'no'}")
    print(f"  Entity docs field: {schema['entity_docs_field'] or '(none)'}")
    print(f"  Entity first-seen field: {schema['entity_first_seen_field'] or '(none)'}")
    print(f"  Relationship doc field: {schema['rel_doc_field'] or '(none)'}")

    # Build chunk/file mapping
    chunk_file_map = {}
    if CHUNK_DB and Path(CHUNK_DB).exists():
        chunk_file_map = build_chunk_to_file_map(CHUNK_DB)
        if not chunk_file_map:
            # Fallback: try chapter-based mapping from chunk DB
            print("  Trying chapter-based chunk mapping...")
            chunk_file_map = build_chunk_to_chapter_map(CHUNK_DB)
    
    if not chunk_file_map:
        # Last resort: try chunks table in the entity DB itself
        if schema['has_chunks_table']:
            print("  Trying chunk mapping from entity DB...")
            chunk_file_map = build_chunk_to_chapter_map(cursor)

    print(f"  Chunk map: {len(chunk_file_map)} entries")

    # ------------------------------------------------------------------
    # NODES (with deduplication by name)
    # ------------------------------------------------------------------
    print("\nExporting entities as nodes...")
    cursor.execute('SELECT * FROM entities')
    entities = cursor.fetchall()

    docs_field = schema['entity_docs_field']           # 'documents_appeared' or 'chunks_appeared' or None
    first_seen_field = schema['entity_first_seen_field']  # 'first_seen_document' or 'first_seen_chunk' or None

    # First pass: group by name
    entities_by_name = {}

    for entity in entities:
        name = safe_get(entity, 'name', 'unknown')
        entity_type = safe_get(entity, 'type', 'unknown')
        description = safe_get(entity, 'description', '')

        # Parse doc/chunk appearances
        raw_appeared = safe_get(entity, docs_field, '[]') if docs_field else '[]'
        try:
            appeared_ids = json.loads(raw_appeared) if raw_appeared else []
        except (json.JSONDecodeError, TypeError):
            appeared_ids = []

        # Resolve first-seen
        first_seen_id = str(safe_get(entity, first_seen_field, '')) if first_seen_field else ''
        first_seen_info = chunk_file_map.get(first_seen_id, {})
        first_seen_path = first_seen_info.get('filename', first_seen_id or 'unknown')
        first_seen_page = first_seen_info.get('page', '?')

        # Build document entries (deduplicated by filename)
        doc_entries = []
        seen_files = set()
        for doc_id in appeared_ids:
            info = chunk_file_map.get(str(doc_id), {})
            fname = info.get('filename', str(doc_id))
            page = info.get('page', '?')
            if fname not in seen_files:
                seen_files.add(fname)
                doc_entries.append({'filename': fname, 'page': page})

        if name in entities_by_name:
            existing = entities_by_name[name]
            existing_fnames = {e['filename'] for e in existing['doc_entries']}
            for de in doc_entries:
                if de['filename'] not in existing_fnames:
                    existing['doc_entries'].append(de)
                    existing_fnames.add(de['filename'])
            if len(doc_entries) > existing.get('_orig_doc_count', 0):
                existing['entity_type'] = entity_type
                existing['description'] = description
                existing['first_seen_path'] = first_seen_path
                existing['first_seen_page'] = first_seen_page
                existing['_orig_doc_count'] = len(doc_entries)
            print(f"  ⚠ Merged duplicate: '{name}' ({existing['entity_type']}+{entity_type})")
        else:
            entities_by_name[name] = {
                'entity_type': entity_type,
                'description': description,
                'first_seen_path': first_seen_path,
                'first_seen_page': first_seen_page,
                'doc_entries': doc_entries,
                '_orig_doc_count': len(doc_entries),
            }

    # Second pass: build node list
    nodes = []
    node_ids = set()

    for name, info in entities_by_name.items():
        doc_entries = info['doc_entries']
        first_seen_path = info['first_seen_path']
        first_seen_page = info['first_seen_page']
        entity_type = info['entity_type']
        description = info['description']

        search_text = build_search_text(cursor, name, entity_type, description, schema)

        first_seen_full = DOC_BASE_PATH + (doc_entries[0]['filename'] if doc_entries else first_seen_path)

        nodes.append({
            'id': name,
            'name': name,
            'entity_type': entity_type,
            'description': description,
            'document_count': len(doc_entries),
            'first_seen_document': first_seen_full,
            'first_seen_display': Path(first_seen_path).name,
            'first_seen_page': first_seen_page,
            'documents': [
                {'filename': Path(e['filename']).name, 'page': e['page'],
                 'path': DOC_BASE_PATH + e['filename']}
                for e in doc_entries
            ],
            'documents_total': len(doc_entries),
            'searchText': search_text,
            'connections': 0
        })
        node_ids.add(name)

    print(f"  {len(nodes)} nodes ({len(entities)} rows before dedup)")

    # ------------------------------------------------------------------
    # EDGES
    # ------------------------------------------------------------------
    print("Exporting relationships as edges...")
    cursor.execute('SELECT * FROM relationships')
    relationships = cursor.fetchall()

    rel_doc_field = schema['rel_doc_field']  # 'document_id' or 'chunk_id' or None

    edge_map = {}
    for rel in relationships:
        source = safe_get(rel, 'source', '')
        target = safe_get(rel, 'target', '')
        if not source or not target:
            continue
        if source not in node_ids or target not in node_ids:
            continue

        key = tuple(sorted([source, target]))

        if key not in edge_map:
            edge_map[key] = {
                'source': key[0],
                'target': key[1],
                'relationships': [],
                'weight': 0
            }

        edge_map[key]['relationships'].append({
            'type': safe_get(rel, 'type', 'unknown'),
            'description': safe_get(rel, 'description', ''),
            'document_id': str(safe_get(rel, rel_doc_field, '?')) if rel_doc_field else '?'
        })
        edge_map[key]['weight'] += 1

    links = list(edge_map.values())

    # Trim relationship details for JSON size
    for link in links:
        if len(link['relationships']) > 5:
            total = len(link['relationships'])
            link['relationships'] = link['relationships'][:5]
            link['relationships_total'] = total
        else:
            link['relationships_total'] = len(link['relationships'])

    print(f"  {sum(e['weight'] for e in links)} relationships across {len(links)} unique edges")

    # ------------------------------------------------------------------
    # CONNECTION COUNTS
    # ------------------------------------------------------------------
    degree = {}
    for link in links:
        degree[link['source']] = degree.get(link['source'], 0) + link['weight']
        degree[link['target']] = degree.get(link['target'], 0) + link['weight']

    for node in nodes:
        node['connections'] = degree.get(node['id'], 0)

    # ------------------------------------------------------------------
    # COMMUNITY DETECTION (Louvain)
    # ------------------------------------------------------------------
    print("\nRunning community detection (Louvain)...")
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        # Build NetworkX graph from our edges
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'])
        for link in links:
            G.add_edge(link['source'], link['target'], weight=link['weight'])

        # Run Louvain
        communities = louvain_communities(G, weight='weight', resolution=1.0, seed=42)

        # Assign community IDs to nodes
        node_community = {}
        for i, community in enumerate(communities):
            for node_id in community:
                node_community[node_id] = i

        for node in nodes:
            node['community'] = node_community.get(node['id'], -1)

        # Build community summary: top entities per community
        community_info = {}
        for i, community in enumerate(communities):
            members = [(nid, degree.get(nid, 0)) for nid in community]
            members.sort(key=lambda x: -x[1])
            top_names = [m[0] for m in members[:5]]
            community_info[i] = {
                'size': len(community),
                'top_entities': top_names
            }

        num_communities = len(communities)
        print(f"  Found {num_communities} communities")
        for i in sorted(community_info.keys(), key=lambda x: -community_info[x]['size'])[:10]:
            info = community_info[i]
            top_str = ', '.join(info['top_entities'][:3])
            print(f"    Community {i}: {info['size']} nodes — {top_str}")

    except ImportError:
        print("  ⚠ networkx not installed — skipping community detection")
        print("    Install with: pip install networkx")
        for node in nodes:
            node['community'] = 0
        num_communities = 1
        community_info = {}
    except Exception as e:
        print(f"  ⚠ Community detection failed: {e}")
        for node in nodes:
            node['community'] = 0
        num_communities = 1
        community_info = {}

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------
    type_counts = {}
    for node in nodes:
        t = node['entity_type']
        type_counts[t] = type_counts.get(t, 0) + 1

    top_by_degree = sorted(nodes, key=lambda n: n['connections'], reverse=True)[:10]

    print(f"\nEntity Types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    print(f"\nMost Connected:")
    for n in top_by_degree:
        print(f"  {n['name']} ({n['entity_type']}): {n['connections']}")

    # ------------------------------------------------------------------
    # WRITE JSON
    # ------------------------------------------------------------------
    output = {
        'nodes': nodes,
        'links': links,
        'stats': {
            'node_count': len(nodes),
            'edge_count': len(links),
            'relationship_count': sum(e['weight'] for e in links),
            'type_counts': type_counts,
            'num_communities': num_communities,
            'communities': {str(k): v for k, v in community_info.items()}
        }
    }

    print(f"\nWriting {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)

    file_size = Path(OUTPUT_JSON).stat().st_size
    print(f"  {file_size / 1024:.0f} KB")
    print(f"\n✓ Export complete: {OUTPUT_JSON}")
    print(f"  Open correspondence_graph_3d.html in your browser")

    conn.close()
    return 0


if __name__ == "__main__":
    exit(export_graph())
