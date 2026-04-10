"""
test_embed_match_entities.py
============================
Tests for embed_match_entities.py (Phase 4 entity deduplication).

Uses synthetic embeddings to test the dedup logic without requiring
a network connection or model download. The embedding model itself
is trusted (sentence-transformers is well-tested); what we need to
verify is that the pipeline correctly finds pairs, clusters them,
and rewrites the database.
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from embed_match_entities import (
    build_entity_text,
    find_candidate_pairs,
    cluster_pairs,
    generate_report,
    UnionFind,
)
from normalize_entities import rewrite_database


# ---------------------------------------------------------------------------
# Test data — entities that Phase 1 and 2 would NOT catch
# ---------------------------------------------------------------------------

ENTITIES = [
    # Group 1: Acronym vs full name — same org
    (1, "EPA",
     "organization", "United States Environmental Protection Agency",
     "doc1", '["doc1","doc2","doc3"]'),
    (2, "Environmental Protection Agency",
     "organization", "Federal environmental regulatory agency",
     "doc4", '["doc4","doc5"]'),

    # Group 2: Role-based reference vs name — same person
    (3, "Robert Chen",
     "person", "Senior structural engineer at Baytown Construction",
     "doc1", '["doc1","doc2","doc3"]'),
    (4, "Lead Structural Engineer",
     "person", "Senior structural engineer responsible for Baytown site",
     "doc4", '["doc4"]'),

    # Group 3: Regulation shorthand vs full title
    (5, "29 CFR 1926.451",
     "regulation", "OSHA scaffolding safety standard requirements",
     "doc1", '["doc1","doc2"]'),
    (6, "OSHA Scaffolding Standard",
     "regulation", "Federal scaffolding safety regulation 29 CFR 1926",
     "doc3", '["doc3"]'),

    # Singleton: should NOT match anyone
    (7, "Jane Williams",
     "person", "Electrical engineer at Pacific Power",
     "doc1", '["doc1","doc2","doc3"]'),

    # Another singleton
    (8, "Baytown Construction",
     "organization", "General contractor for the Baytown development project",
     "doc1", '["doc1","doc2"]'),

    # Group 4: Same project, different naming
    (9, "Phase 3 Foundation Work",
     "project", "Foundation construction for Phase 3 of Baytown development",
     "doc1", '["doc1","doc2"]'),
    (10, "Baytown Phase 3 Foundations",
     "project", "Foundation work on Phase 3 of the Baytown construction project",
     "doc3", '["doc3"]'),
]

RELATIONSHIPS = [
    (1, "Robert Chen",            "Baytown Construction", "works_for",  "engineer at Baytown",  "doc1"),
    (2, "Lead Structural Engineer","Baytown Construction", "works_for",  "works for Baytown",    "doc4"),
    (3, "EPA",                    "Baytown Construction", "inspected",  "EPA inspection",       "doc1"),
    (4, "Environmental Protection Agency", "Baytown Construction", "inspected","agency inspection","doc4"),
    (5, "Robert Chen",            "29 CFR 1926.451",      "violated",   "scaffolding violation", "doc1"),
    (6, "Lead Structural Engineer","OSHA Scaffolding Standard","violated","safety violation",     "doc4"),
]

DOCUMENTS = [
    ("doc1", "Robert Chen", "EPA", None, None, "2024-01-15", "Inspection Report", "REF-001", "f1.pdf"),
    ("doc4", "Lead Structural Engineer", "Environmental Protection Agency", None, None, "2024-03-10", "Response", "REF-002", "f2.pdf"),
]


def make_entities():
    """Convert ENTITIES tuples to list of dicts."""
    return [dict(zip(
        ["entity_id", "name", "type", "description",
         "first_seen_document", "documents_appeared"],
        row,
    )) for row in ENTITIES]


def make_synthetic_embeddings(entities):
    """
    Build synthetic embeddings that encode the expected similarity structure.

    Entities in the same group get nearly-identical embeddings (cosine ~ 0.95).
    Singletons and different groups get orthogonal/distant embeddings.
    """
    n = len(entities)
    dim = 64
    rng = np.random.RandomState(42)

    # Start with random embeddings
    embeddings = rng.randn(n, dim).astype(np.float32)

    # Group 1: entities 0, 1 (EPA / Environmental Protection Agency)
    base = rng.randn(dim).astype(np.float32)
    embeddings[0] = base + rng.randn(dim) * 0.08
    embeddings[1] = base + rng.randn(dim) * 0.08

    # Group 2: entities 2, 3 (Robert Chen / Lead Structural Engineer)
    base = rng.randn(dim).astype(np.float32)
    embeddings[2] = base + rng.randn(dim) * 0.08
    embeddings[3] = base + rng.randn(dim) * 0.08

    # Group 3: entities 4, 5 (29 CFR / OSHA Scaffolding)
    base = rng.randn(dim).astype(np.float32)
    embeddings[4] = base + rng.randn(dim) * 0.08
    embeddings[5] = base + rng.randn(dim) * 0.08

    # Group 4: entities 8, 9 (Phase 3 Foundation Work / Baytown Phase 3)
    base = rng.randn(dim).astype(np.float32)
    embeddings[8] = base + rng.randn(dim) * 0.08
    embeddings[9] = base + rng.randn(dim) * 0.08

    # Singletons: entities 6, 7 — keep random (distant from groups)

    # L2-normalize all embeddings (so dot product = cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return embeddings


def create_test_db(db_path):
    """Create a synthetic entity database for Phase 4 testing."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE entities (
        entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, type TEXT NOT NULL, description TEXT,
        first_seen_document TEXT, documents_appeared TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(name, type)
    )''')
    c.execute('''CREATE TABLE relationships (
        relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL, target TEXT NOT NULL, type TEXT NOT NULL,
        description TEXT, document_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE documents (
        document_id TEXT PRIMARY KEY, sender TEXT, recipient TEXT,
        cc TEXT, bcc TEXT, date TEXT, subject TEXT, reference TEXT,
        source_file TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE processing_log (
        document_id TEXT PRIMARY KEY, processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        success BOOLEAN, error_message TEXT
    )''')

    for eid, name, etype, desc, first_doc, docs in ENTITIES:
        c.execute(
            "INSERT INTO entities (entity_id, name, type, description, "
            "first_seen_document, documents_appeared) VALUES (?,?,?,?,?,?)",
            (eid, name, etype, desc, first_doc, docs),
        )
    for rid, src, tgt, rtype, desc, doc_id in RELATIONSHIPS:
        c.execute(
            "INSERT INTO relationships (relationship_id, source, target, type, "
            "description, document_id) VALUES (?,?,?,?,?,?)",
            (rid, src, tgt, rtype, desc, doc_id),
        )
    for row in DOCUMENTS:
        c.execute(
            "INSERT INTO documents (document_id, sender, recipient, cc, bcc, "
            "date, subject, reference, source_file) VALUES (?,?,?,?,?,?,?,?,?)",
            row,
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildEntityText(unittest.TestCase):
    """Test composite text construction."""

    def test_full_entity(self):
        text = build_entity_text({
            "name": "John Doe", "type": "person", "description": "Site inspector"
        })
        self.assertEqual(text, "John Doe (person): Site inspector")

    def test_no_description(self):
        text = build_entity_text({
            "name": "ACME", "type": "organization", "description": None
        })
        self.assertEqual(text, "ACME (organization):")

    def test_empty_description(self):
        text = build_entity_text({
            "name": "ACME", "type": "organization", "description": ""
        })
        self.assertEqual(text, "ACME (organization):")


class TestSyntheticEmbeddings(unittest.TestCase):
    """Verify our synthetic embeddings have the expected similarity structure."""

    def setUp(self):
        self.entities = make_entities()
        self.embeddings = make_synthetic_embeddings(self.entities)

    def test_shape(self):
        self.assertEqual(self.embeddings.shape[0], len(self.entities))

    def test_normalized(self):
        norms = np.linalg.norm(self.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_group_similarity_high(self):
        """Entities in the same group should have high cosine similarity."""
        groups = [(0, 1), (2, 3), (4, 5), (8, 9)]
        for i, j in groups:
            sim = float(self.embeddings[i] @ self.embeddings[j])
            self.assertGreater(sim, 0.80,
                f"Entities {i} ({self.entities[i]['name']}) and "
                f"{j} ({self.entities[j]['name']}) have low sim: {sim:.3f}")

    def test_singleton_similarity_low(self):
        """Singletons should have low similarity to non-group entities."""
        singleton_idx = 6  # Jane Williams
        for i in range(len(self.entities)):
            if i == singleton_idx:
                continue
            sim = float(self.embeddings[singleton_idx] @ self.embeddings[i])
            self.assertLess(sim, 0.80,
                f"Singleton {self.entities[singleton_idx]['name']} has high sim "
                f"with {self.entities[i]['name']}: {sim:.3f}")


class TestFindCandidatePairs(unittest.TestCase):
    """Test that candidate pair finding works with synthetic embeddings."""

    def setUp(self):
        self.entities = make_entities()
        self.embeddings = make_synthetic_embeddings(self.entities)

    def test_finds_epa_pair(self):
        """EPA and Environmental Protection Agency should match."""
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.80, 10)
        pair_names = set()
        for a, b, _ in pairs:
            pair_names.add((a["name"], b["name"]))
            pair_names.add((b["name"], a["name"]))
        self.assertIn(("EPA", "Environmental Protection Agency"), pair_names)

    def test_finds_person_pair(self):
        """Robert Chen and Lead Structural Engineer should match."""
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.80, 10)
        pair_names = set()
        for a, b, _ in pairs:
            pair_names.add((a["name"], b["name"]))
            pair_names.add((b["name"], a["name"]))
        self.assertIn(("Robert Chen", "Lead Structural Engineer"), pair_names)

    def test_finds_regulation_pair(self):
        """29 CFR 1926.451 and OSHA Scaffolding Standard should match."""
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.80, 10)
        pair_names = set()
        for a, b, _ in pairs:
            pair_names.add((a["name"], b["name"]))
            pair_names.add((b["name"], a["name"]))
        self.assertIn(("29 CFR 1926.451", "OSHA Scaffolding Standard"), pair_names)

    def test_singleton_not_paired(self):
        """Jane Williams should not appear in pairs at a reasonable threshold."""
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.80, 10)
        names_in_pairs = set()
        for a, b, _ in pairs:
            names_in_pairs.add(a["name"])
            names_in_pairs.add(b["name"])
        self.assertNotIn("Jane Williams", names_in_pairs)

    def test_pairs_sorted_descending(self):
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.70, 10)
        scores = [s["cosine_sim"] for _, _, s in pairs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_skips_same_normalized_key(self):
        """Pairs with identical normalized names should be skipped."""
        # Add entity "epa" which normalizes same as "EPA"
        extra = self.entities + [
            {"entity_id": 99, "name": "epa", "type": "organization",
             "description": "Environmental agency", "first_seen_document": "doc9",
             "documents_appeared": '["doc9"]'},
        ]
        # Append a copy of embedding 0 (high sim to "EPA")
        extra_emb = np.vstack([self.embeddings, self.embeddings[0:1]])
        pairs = find_candidate_pairs(extra, extra_emb, 0.5, 10)
        for a, b, _ in pairs:
            if {a["name"], b["name"]} == {"EPA", "epa"}:
                self.fail("Pair with identical normalized key should be skipped")


class TestClusterPairs(unittest.TestCase):
    """Test union-find clustering."""

    def setUp(self):
        self.entities = make_entities()
        self.embeddings = make_synthetic_embeddings(self.entities)

    def test_produces_groups(self):
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.80, 10)
        groups, _ = cluster_pairs(pairs, self.entities)
        self.assertGreater(len(groups), 0)

    def test_all_entities_in_name_map(self):
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.80, 10)
        _, name_map = cluster_pairs(pairs, self.entities)
        for ent in self.entities:
            self.assertIn(ent["name"], name_map)

    def test_canonical_has_most_docs(self):
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.80, 10)
        groups, _ = cluster_pairs(pairs, self.entities)
        for mg in groups:
            canonical = mg["canonical"]
            try:
                canon_docs = len(json.loads(canonical["documents_appeared"])) if canonical["documents_appeared"] else 0
            except (json.JSONDecodeError, TypeError):
                canon_docs = 0
            for member in mg["members"]:
                try:
                    member_docs = len(json.loads(member["documents_appeared"])) if member["documents_appeared"] else 0
                except (json.JSONDecodeError, TypeError):
                    member_docs = 0
                self.assertGreaterEqual(canon_docs, member_docs)

    def test_epa_canonical_selection(self):
        """EPA (3 docs) should be canonical over Environmental Protection Agency (2 docs)."""
        pairs = find_candidate_pairs(self.entities, self.embeddings, 0.80, 10)
        _, name_map = cluster_pairs(pairs, self.entities)
        self.assertEqual(name_map["Environmental Protection Agency"], "EPA")


class TestFullPipeline(unittest.TestCase):
    """Test the full DB rewrite using embedding matching."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.source_db = os.path.join(self.tmpdir, "source.db")
        self.output_db = os.path.join(self.tmpdir, "output.db")
        create_test_db(self.source_db)

        conn = sqlite3.connect(self.source_db)
        conn.row_factory = sqlite3.Row
        entities = [dict(row) for row in conn.execute("SELECT * FROM entities").fetchall()]
        conn.close()

        embeddings = make_synthetic_embeddings(entities)
        pairs = find_candidate_pairs(entities, embeddings, 0.80, 10)
        self.merge_groups, self.name_map = cluster_pairs(pairs, entities)
        self.stats = rewrite_database(
            self.source_db, self.output_db, self.name_map, self.merge_groups
        )

    def tearDown(self):
        for f in Path(self.tmpdir).glob("*"):
            f.unlink()
        os.rmdir(self.tmpdir)

    def test_entity_count_reduced(self):
        self.assertLess(self.stats["final_entity_count"], len(ENTITIES))

    def test_canonical_names_in_output(self):
        conn = sqlite3.connect(self.output_db)
        names = {row[0] for row in conn.execute("SELECT name FROM entities").fetchall()}
        conn.close()
        self.assertIn("EPA", names)
        self.assertIn("Robert Chen", names)
        self.assertNotIn("Environmental Protection Agency", names)
        self.assertNotIn("Lead Structural Engineer", names)

    def test_relationships_rewritten(self):
        conn = sqlite3.connect(self.output_db)
        names_in_rels = set()
        for row in conn.execute("SELECT source, target FROM relationships"):
            names_in_rels.add(row[0])
            names_in_rels.add(row[1])
        conn.close()
        self.assertNotIn("Environmental Protection Agency", names_in_rels)
        self.assertNotIn("Lead Structural Engineer", names_in_rels)
        self.assertIn("EPA", names_in_rels)
        self.assertIn("Robert Chen", names_in_rels)

    def test_documents_table_updated(self):
        conn = sqlite3.connect(self.output_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT sender, recipient FROM documents").fetchall()
        conn.close()
        all_vals = set()
        for row in rows:
            if row["sender"]:
                all_vals.add(row["sender"])
            if row["recipient"]:
                all_vals.add(row["recipient"])
        self.assertNotIn("Environmental Protection Agency", all_vals)
        self.assertNotIn("Lead Structural Engineer", all_vals)


class TestHTMLReport(unittest.TestCase):
    def test_report_generation(self):
        entities = make_entities()
        embeddings = make_synthetic_embeddings(entities)
        pairs = find_candidate_pairs(entities, embeddings, 0.80, 10)
        groups, name_map = cluster_pairs(pairs, entities)

        html = generate_report(
            pairs, groups, name_map,
            "test_in.db", "test_out.db",
            len(entities), 6,
            {"final_entity_count": 7, "final_rel_count": 5,
             "relationships_rewritten": 3, "relationships_deduped": 1},
            0.82, "all-MiniLM-L6-v2", 1.5,
        )

        self.assertIn("Embedding Match Report", html)
        self.assertIn("const PAIRS", html)
        self.assertIn("const GROUPS", html)
        self.assertIn("cosine", html)


class TestUnionFind(unittest.TestCase):
    def test_basic(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        self.assertEqual(uf.find("a"), uf.find("c"))

    def test_separate(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.find("c")
        self.assertNotEqual(uf.find("a"), uf.find("c"))

    def test_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.find("d")
        comps = uf.components()
        self.assertEqual(len(comps), 2)


if __name__ == "__main__":
    unittest.main()
