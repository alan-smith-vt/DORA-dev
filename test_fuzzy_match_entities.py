"""
test_fuzzy_match_entities.py
============================
Tests for fuzzy_match_entities.py (Phase 2 entity deduplication).
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fuzzy_match_entities import (
    score_pair,
    tokenize_for_blocking,
    build_candidate_pairs,
    cluster_pairs,
    generate_report,
    UnionFind,
)
from normalize_entities import normalize_name, rewrite_database


# ---------------------------------------------------------------------------
# Test data — entities that Phase 1 normalization would NOT catch
# (different normalized keys, but clearly the same entity via fuzzy match)
# ---------------------------------------------------------------------------

ENTITIES = [
    # Group 1: Typo in first name — "Jon" vs "John"
    (1, "John Doe",    "person",       "Site inspector",        "doc1", '["doc1","doc2","doc3"]'),
    (2, "Jon Doe",     "person",       "Inspector at site",     "doc4", '["doc4"]'),

    # Group 2: Abbreviated first name — "R. Smith" vs "Robert Smith"
    (3, "Robert Smith",  "person",     "Project manager",       "doc1", '["doc1","doc2"]'),
    (4, "R. Smith",      "person",     "PM",                    "doc3", '["doc3"]'),

    # Group 3: Org name variation — "Baytown Construction" vs "Baytown Const"
    (5, "Baytown Construction",   "organization", "General contractor",  "doc1", '["doc1","doc2","doc3"]'),
    (6, "Baytown Const",          "organization", "Contractor",          "doc5", '["doc5"]'),

    # Singleton: should NOT fuzzy-match anyone
    (7, "Jane Williams", "person",     "Engineer",              "doc1", '["doc1","doc2"]'),

    # Group 4: Swapped order — "Smith, William" vs "William Smith"
    (8, "William Smith",   "person",   "Welder",                "doc1", '["doc1","doc2"]'),
    (9, "Smith, William",  "person",   "Site welder",           "doc3", '["doc3"]'),

    # Anti-pair: similar but genuinely different people
    (10, "James Wilson",    "person",  "Electrical engineer",   "doc1", '["doc1"]'),
    (11, "James Williams",  "person",  "Structural engineer",   "doc2", '["doc2"]'),

    # Group 5: Extra middle initial
    (12, "Michael Johnson",    "person", "Inspector",           "doc1", '["doc1","doc2"]'),
    (13, "Michael A. Johnson", "person", "Lead inspector",      "doc3", '["doc3"]'),
]

RELATIONSHIPS = [
    (1, "John Doe",             "Baytown Construction", "works_for",   "works for Baytown",     "doc1"),
    (2, "Jon Doe",              "Baytown Const",        "works_for",   "works for Baytown",     "doc4"),
    (3, "Robert Smith",         "Baytown Construction", "inspected",   "inspected the site",    "doc1"),
    (4, "R. Smith",             "Baytown Construction", "inspected",   "site inspection",       "doc3"),
    (5, "William Smith",        "Baytown Construction", "works_for",   "welder at Baytown",     "doc1"),
    (6, "Smith, William",       "Baytown Const",        "works_for",   "welder",                "doc3"),
    (7, "Michael Johnson",      "Jane Williams",        "documented",  "filed report",          "doc1"),
    (8, "Michael A. Johnson",   "Jane Williams",        "documented",  "filed report",          "doc3"),
]

DOCUMENTS = [
    ("doc1", "John Doe",    "Robert Smith", None, None, "2024-01-15", "Report", "REF-001", "f1.pdf"),
    ("doc3", "R. Smith",    "Jon Doe",      None, None, "2024-02-01", "Follow-up", "REF-002", "f2.pdf"),
    ("doc4", "Jon Doe",     "Baytown Const", None, None, "2024-03-10", "Response", "REF-003", "f3.pdf"),
]


def create_test_db(db_path):
    """Create a synthetic entity database for Phase 2 testing."""
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

class TestScorePair(unittest.TestCase):
    """Test the similarity scoring function."""

    def test_identical(self):
        scores = score_pair("john doe", "john doe")
        self.assertAlmostEqual(scores["combined"], 1.0, places=3)

    def test_typo(self):
        """'jon doe' vs 'john doe' should score high on Jaro-Winkler."""
        scores = score_pair("jon doe", "john doe")
        self.assertGreater(scores["jaro_winkler"], 0.90)
        self.assertGreater(scores["combined"], 0.90)

    def test_token_reorder(self):
        """'smith william' vs 'william smith' should have high token-sort ratio."""
        scores = score_pair("smith william", "william smith")
        self.assertGreater(scores["token_sort_ratio"], 0.95)

    def test_abbreviation(self):
        """'r smith' vs 'robert smith' — token-sort is moderate (not inflated)."""
        scores = score_pair("r smith", "robert smith")
        self.assertGreater(scores["token_sort_ratio"], 0.60)
        # Crucially, token_sort does NOT give 100% for subset matches
        self.assertLess(scores["token_sort_ratio"], 0.90)

    def test_unrelated_low_score(self):
        """Completely different names should score low."""
        scores = score_pair("john doe", "jane williams")
        self.assertLess(scores["combined"], 0.80)

    def test_subset_not_inflated(self):
        """Token-sort should NOT give 100% for subset matches.

        This was the core bug with token-set ratio: 'michael johnson' is a
        token subset of 'michael a johnson', so token-set gave 100%.
        Token-sort correctly gives a lower score.
        """
        scores = score_pair("michael johnson", "michael a johnson")
        self.assertLess(scores["token_sort_ratio"], 1.0)
        # Combined should still be high (JW catches it) but not 100%
        self.assertLess(scores["combined"], 1.0)
        self.assertGreater(scores["combined"], 0.90)

    def test_similar_but_different(self):
        """'james wilson' vs 'james williams' — similar but probably different people."""
        scores = score_pair("james wilson", "james williams")
        # Should be highish but not above a conservative threshold
        self.assertLess(scores["combined"], 0.95)


class TestTokenBlocking(unittest.TestCase):
    """Test the token-blocking index."""

    def test_shared_token_generates_pair(self):
        """Entities sharing a non-stop token should become candidates."""
        tokens_a = tokenize_for_blocking("john doe")
        tokens_b = tokenize_for_blocking("jon doe")
        # "doe" is shared
        self.assertTrue(tokens_a & tokens_b)

    def test_stop_words_excluded(self):
        """Stop words should not be blocking keys."""
        tokens = tokenize_for_blocking("the project for site inspection")
        self.assertNotIn("the", tokens)
        self.assertNotIn("for", tokens)

    def test_short_tokens_excluded(self):
        """Single-char tokens should be excluded."""
        tokens = tokenize_for_blocking("j smith")
        self.assertNotIn("j", tokens)
        self.assertIn("smith", tokens)


class TestBuildCandidatePairs(unittest.TestCase):
    """Test candidate pair generation with scoring."""

    def setUp(self):
        self.entities = [dict(zip(
            ["entity_id", "name", "type", "description",
             "first_seen_document", "documents_appeared"],
            row,
        )) for row in ENTITIES]

    def test_finds_typo_pair(self):
        """Should find John Doe / Jon Doe pair."""
        pairs = build_candidate_pairs(self.entities, threshold=0.88)
        pair_names = {(a["name"], b["name"]) for a, b, _ in pairs}
        pair_names |= {(b, a) for a, b in pair_names}
        self.assertIn(("John Doe", "Jon Doe"), pair_names)

    def test_finds_reorder_pair(self):
        """Should find William Smith / Smith, William pair."""
        pairs = build_candidate_pairs(self.entities, threshold=0.88)
        pair_names = {(a["name"], b["name"]) for a, b, _ in pairs}
        pair_names |= {(b, a) for a, b in pair_names}
        self.assertIn(("William Smith", "Smith, William"), pair_names)

    def test_singleton_not_paired(self):
        """Jane Williams should not appear in any pair at a strict threshold."""
        # At 0.95, "Jane Williams" vs "James Williams" (JW=0.94) is excluded
        pairs = build_candidate_pairs(self.entities, threshold=0.95)
        names_in_pairs = set()
        for a, b, _ in pairs:
            names_in_pairs.add(a["name"])
            names_in_pairs.add(b["name"])
        self.assertNotIn("Jane Williams", names_in_pairs)

    def test_scores_sorted_descending(self):
        """Pairs should be sorted by combined score descending."""
        pairs = build_candidate_pairs(self.entities, threshold=0.80)
        scores = [s["combined"] for _, _, s in pairs]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestUnionFind(unittest.TestCase):
    """Test the union-find data structure."""

    def test_basic_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        self.assertEqual(uf.find("a"), uf.find("c"))

    def test_separate_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        self.assertNotEqual(uf.find("a"), uf.find("c"))

    def test_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.find("d")  # singleton
        comps = uf.components()
        # a, b, c in one component; d alone
        self.assertEqual(len(comps), 2)


class TestClusterPairs(unittest.TestCase):
    """Test clustering pairs into merge groups."""

    def setUp(self):
        self.entities = [dict(zip(
            ["entity_id", "name", "type", "description",
             "first_seen_document", "documents_appeared"],
            row,
        )) for row in ENTITIES]

    def test_clustering_produces_groups(self):
        pairs = build_candidate_pairs(self.entities, threshold=0.88)
        groups, name_map = cluster_pairs(pairs, self.entities)
        self.assertGreater(len(groups), 0)

    def test_canonical_is_most_documented(self):
        """Canonical should be the member with the most document appearances."""
        pairs = build_candidate_pairs(self.entities, threshold=0.88)
        groups, name_map = cluster_pairs(pairs, self.entities)

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
                self.assertGreaterEqual(canon_docs, member_docs,
                    f"Canonical {canonical['name']} ({canon_docs} docs) has fewer docs "
                    f"than member {member['name']} ({member_docs} docs)")

    def test_all_entities_in_name_map(self):
        pairs = build_candidate_pairs(self.entities, threshold=0.88)
        _, name_map = cluster_pairs(pairs, self.entities)
        for ent in self.entities:
            self.assertIn(ent["name"], name_map)


class TestFullPipeline(unittest.TestCase):
    """Test the full DB rewrite using fuzzy matching."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.source_db = os.path.join(self.tmpdir, "source.db")
        self.output_db = os.path.join(self.tmpdir, "output.db")
        create_test_db(self.source_db)

        conn = sqlite3.connect(self.source_db)
        conn.row_factory = sqlite3.Row
        entities = [dict(row) for row in conn.execute("SELECT * FROM entities").fetchall()]
        conn.close()

        pairs = build_candidate_pairs(entities, threshold=0.88)
        self.merge_groups, self.name_map = cluster_pairs(pairs, entities)
        self.stats = rewrite_database(
            self.source_db, self.output_db, self.name_map, self.merge_groups
        )

    def tearDown(self):
        for f in Path(self.tmpdir).glob("*"):
            f.unlink()
        os.rmdir(self.tmpdir)

    def test_entity_count_reduced(self):
        """Output should have fewer entities."""
        self.assertLess(
            self.stats["final_entity_count"],
            len(ENTITIES),
            "Entity count should be reduced by fuzzy merging",
        )

    def test_canonical_names_in_output(self):
        """Canonical names should exist in output."""
        conn = sqlite3.connect(self.output_db)
        names = {row[0] for row in conn.execute("SELECT name FROM entities").fetchall()}
        conn.close()

        # John Doe (3 docs) should be canonical over Jon Doe (1 doc)
        self.assertIn("John Doe", names)
        self.assertNotIn("Jon Doe", names)

    def test_relationships_use_canonical_names(self):
        """No non-canonical names should appear in relationships."""
        conn = sqlite3.connect(self.output_db)

        # "Jon Doe" should be rewritten to "John Doe"
        rows = conn.execute(
            "SELECT * FROM relationships WHERE source = 'Jon Doe' OR target = 'Jon Doe'"
        ).fetchall()
        self.assertEqual(len(rows), 0, "'Jon Doe' still appears in relationships")

        conn.close()

    def test_documents_table_updated(self):
        """Sender/recipient in documents table should use canonical names."""
        conn = sqlite3.connect(self.output_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT sender, recipient FROM documents").fetchall()
        conn.close()

        all_values = set()
        for row in rows:
            if row["sender"]:
                all_values.add(row["sender"])
            if row["recipient"]:
                all_values.add(row["recipient"])

        # Non-canonical names should be rewritten
        self.assertNotIn("Jon Doe", all_values)


class TestHTMLReport(unittest.TestCase):
    """Test that the HTML report generates correctly."""

    def test_report_generation(self):
        entities = [dict(zip(
            ["entity_id", "name", "type", "description",
             "first_seen_document", "documents_appeared"],
            row,
        )) for row in ENTITIES]

        pairs = build_candidate_pairs(entities, threshold=0.85)
        groups, name_map = cluster_pairs(pairs, entities)

        html = generate_report(
            pairs, groups, name_map,
            "test_input.db", "test_output.db",
            len(entities), 8,
            {"final_entity_count": 8, "final_rel_count": 6,
             "relationships_rewritten": 4, "relationships_deduped": 1},
            0.88, 0.05,
        )

        self.assertIn("Fuzzy Match Report", html)
        self.assertIn("const PAIRS", html)
        self.assertIn("const GROUPS", html)
        self.assertIn("Candidate Pairs", html)
        # Should reference token-sort, not token-set
        self.assertIn("TokenSort", html)
        self.assertNotIn("TokenSet", html)


if __name__ == "__main__":
    unittest.main()
