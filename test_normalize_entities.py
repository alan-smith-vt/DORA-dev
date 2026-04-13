"""
test_normalize_entities.py
==========================
End-to-end test for normalize_entities.py.

Creates a synthetic entity database with known duplicates, runs the
normalization pipeline, and verifies the output database and report.
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent))

from normalize_entities import normalize_name, build_merge_groups, rewrite_database, generate_report


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

ENTITIES = [
    # Group 1: "John Doe" variants (person vs org cross-type)
    (1, "John Doe",       "person",       "Site inspector",        "doc1", '["doc1","doc2","doc3"]'),
    (2, "john doe",        "person",       "Inspector",             "doc2", '["doc2"]'),
    (3, "JOHN DOE",        "organization", "Doe Consulting",        "doc4", '["doc4"]'),
    (4, "Mr. John Doe",    "person",       "Lead inspector",        "doc1", '["doc1"]'),
    (5, "John  Doe",       "person",       "",                      "doc5", '["doc5"]'),

    # Group 2: "ACME Corp" variants
    (6, "ACME Corp",       "organization", "General contractor",    "doc1", '["doc1","doc2"]'),
    (7, "Acme Corp, Inc.", "organization", "The contractor",        "doc3", '["doc3"]'),
    (8, "acme corp",       "organization", "",                      "doc6", '["doc6"]'),

    # Group 3: "U.S. EPA" abbreviation handling
    (9,  "U.S. EPA",       "organization", "Environmental agency",  "doc1", '["doc1"]'),
    (10, "US EPA",         "organization", "Federal agency",        "doc2", '["doc2"]'),

    # Singleton: should not merge with anything
    (11, "Jane Smith",     "person",       "Project manager",       "doc1", '["doc1","doc2","doc3"]'),

    # Group 4: Parenthetical annotation
    (12, "Bob Wilson",               "person", "Welder",            "doc1", '["doc1","doc2"]'),
    (13, "Bob Wilson (Site Lead)",    "person", "Site lead welder",  "doc3", '["doc3"]'),

    # Group 5: Dr. prefix
    (14, "Dr. Sarah Chen",  "person", "Structural engineer",       "doc1", '["doc1","doc2"]'),
    (15, "Sarah Chen",      "person", "Engineer",                   "doc3", '["doc3"]'),
]

RELATIONSHIPS = [
    (1,  "John Doe",    "ACME Corp",    "works_for",       "Doe works for ACME",      "doc1"),
    (2,  "john doe",    "ACME Corp",    "works_for",       "works for ACME",          "doc2"),
    (3,  "John Doe",    "Jane Smith",   "noted",           "Doe noted issue",         "doc1"),
    (4,  "Mr. John Doe","ACME Corp",    "inspected",       "inspection for ACME",     "doc1"),
    (5,  "ACME Corp",   "U.S. EPA",     "violated",        "EPA violation",           "doc1"),
    (6,  "Acme Corp, Inc.", "US EPA",   "violated",        "another violation",       "doc2"),
    (7,  "Bob Wilson",  "ACME Corp",    "works_for",       "welder at ACME",          "doc1"),
    (8,  "Bob Wilson (Site Lead)", "acme corp", "responsible_for", "leads site work",  "doc3"),
    (9,  "Dr. Sarah Chen", "ACME Corp", "inspected",       "structural inspection",   "doc1"),
    (10, "Sarah Chen",  "Jane Smith",   "documented",      "Chen documented report",  "doc3"),
]

DOCUMENTS = [
    ("doc1", "John Doe",  "Jane Smith", None, None, "2024-01-15", "Inspection Report", "REF-001", "file1.pdf"),
    ("doc2", "john doe",  "ACME Corp",  None, None, "2024-02-01", "Follow-up",         "REF-002", "file2.pdf"),
    ("doc3", "Acme Corp, Inc.", "U.S. EPA", None, None, "2024-03-10", "Response",       "REF-003", "file3.pdf"),
]


def create_test_db(db_path):
    """Create a synthetic entity database for testing."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            first_seen_document TEXT,
            documents_appeared TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, type)
        )
    ''')

    c.execute('''
        CREATE TABLE relationships (
            relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            document_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            sender TEXT,
            recipient TEXT,
            cc TEXT,
            bcc TEXT,
            date TEXT,
            subject TEXT,
            reference TEXT,
            source_file TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE processing_log (
            document_id TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN,
            error_message TEXT
        )
    ''')

    # Create indexes to match production schema
    c.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_relationship_source ON relationships(source)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_relationship_target ON relationships(target)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_relationship_document ON relationships(document_id)')

    for eid, name, etype, desc, first_doc, docs in ENTITIES:
        c.execute(
            "INSERT INTO entities (entity_id, name, type, description, first_seen_document, documents_appeared) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (eid, name, etype, desc, first_doc, docs),
        )

    for rid, src, tgt, rtype, desc, doc_id in RELATIONSHIPS:
        c.execute(
            "INSERT INTO relationships (relationship_id, source, target, type, description, document_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rid, src, tgt, rtype, desc, doc_id),
        )

    for row in DOCUMENTS:
        c.execute(
            "INSERT INTO documents (document_id, sender, recipient, cc, bcc, date, subject, reference, source_file) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNormalizeName(unittest.TestCase):
    """Test the normalize_name function in isolation."""

    def test_case_folding(self):
        self.assertEqual(normalize_name("John Doe"), normalize_name("john doe"))
        self.assertEqual(normalize_name("ACME CORP"), normalize_name("acme corp"))

    def test_whitespace_collapse(self):
        self.assertEqual(normalize_name("John  Doe"), normalize_name("John Doe"))
        self.assertEqual(normalize_name("  John Doe  "), normalize_name("John Doe"))

    def test_title_stripping(self):
        self.assertEqual(normalize_name("Mr. John Doe"), normalize_name("John Doe"))
        self.assertEqual(normalize_name("Dr. Sarah Chen"), normalize_name("Sarah Chen"))
        self.assertEqual(normalize_name("Prof. Smith"), normalize_name("Smith"))

    def test_org_suffix_stripping(self):
        self.assertEqual(normalize_name("ACME Corp, Inc."), normalize_name("ACME Corp"))
        self.assertEqual(normalize_name("Foo LLC"), normalize_name("Foo"))
        self.assertEqual(normalize_name("Bar Ltd."), normalize_name("Bar"))

    def test_abbreviation_dots(self):
        self.assertEqual(normalize_name("U.S. EPA"), normalize_name("US EPA"))
        self.assertEqual(normalize_name("U.S.A."), normalize_name("USA"))

    def test_parenthetical_removal(self):
        self.assertEqual(
            normalize_name("Bob Wilson (Site Lead)"),
            normalize_name("Bob Wilson"),
        )

    def test_empty_and_none(self):
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_name(None), "")

    def test_no_false_merges(self):
        # These should NOT normalize to the same thing
        self.assertNotEqual(normalize_name("John Doe"), normalize_name("Jane Doe"))
        self.assertNotEqual(normalize_name("ACME Corp"), normalize_name("APEX Corp"))
        self.assertNotEqual(normalize_name("Jane Smith"), normalize_name("John Smith"))


class TestBuildMergeGroups(unittest.TestCase):
    """Test the merge grouping logic."""

    def setUp(self):
        self.entities = [dict(zip(
            ["entity_id", "name", "type", "description", "first_seen_document", "documents_appeared"],
            row,
        )) for row in ENTITIES]

    def test_groups_found(self):
        groups, name_map = build_merge_groups(self.entities)
        # We expect 5 merge groups (John Doe, ACME, EPA, Bob Wilson, Sarah Chen)
        self.assertEqual(len(groups), 5)

    def test_singleton_not_grouped(self):
        groups, name_map = build_merge_groups(self.entities)
        # Jane Smith should map to herself
        self.assertEqual(name_map["Jane Smith"], "Jane Smith")
        # She should not be in any merge group
        group_canonicals = {g["canonical"]["name"] for g in groups}
        self.assertNotIn("Jane Smith", group_canonicals)

    def test_canonical_selection(self):
        """Canonical should be the entity with most document appearances."""
        groups, name_map = build_merge_groups(self.entities)
        # "John Doe" has 3 docs, others have fewer -> should be canonical
        self.assertEqual(name_map["john doe"], "John Doe")
        self.assertEqual(name_map["JOHN DOE"], "John Doe")
        self.assertEqual(name_map["Mr. John Doe"], "John Doe")

    def test_cross_type_merge(self):
        """Entities with same normalized name but different types should merge."""
        _, name_map = build_merge_groups(self.entities)
        # "JOHN DOE" is type=organization, should merge with "John Doe" type=person
        self.assertEqual(name_map["JOHN DOE"], "John Doe")

    def test_all_names_mapped(self):
        _, name_map = build_merge_groups(self.entities)
        for ent in self.entities:
            self.assertIn(ent["name"], name_map)


class TestRewriteDatabase(unittest.TestCase):
    """Test the full DB rewrite pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.source_db = os.path.join(self.tmpdir, "source.db")
        self.output_db = os.path.join(self.tmpdir, "output.db")
        create_test_db(self.source_db)

        # Build merge info
        conn = sqlite3.connect(self.source_db)
        conn.row_factory = sqlite3.Row
        entities = [dict(row) for row in conn.execute("SELECT * FROM entities").fetchall()]
        conn.close()

        self.merge_groups, self.name_map = build_merge_groups(entities)
        self.stats = rewrite_database(self.source_db, self.output_db, self.name_map, self.merge_groups)

    def tearDown(self):
        for f in Path(self.tmpdir).glob("*"):
            f.unlink()
        os.rmdir(self.tmpdir)

    def test_entity_count_reduced(self):
        """Output should have fewer entities than input."""
        # 15 original entities, 5 groups with multiple members
        # Group 1: 5 members -> 1 (remove 4)
        # Group 2: 3 members -> 1 (remove 2)
        # Group 3: 2 members -> 1 (remove 1)
        # Group 4: 2 members -> 1 (remove 1)
        # Group 5: 2 members -> 1 (remove 1)
        # Total removed: 9, expected: 15 - 9 = 6
        self.assertEqual(self.stats["final_entity_count"], 6)

    def test_canonical_entities_exist(self):
        """Each canonical name should exist in the output."""
        conn = sqlite3.connect(self.output_db)
        c = conn.cursor()
        names = {row[0] for row in c.execute("SELECT name FROM entities").fetchall()}
        conn.close()

        self.assertIn("John Doe", names)
        self.assertIn("ACME Corp", names)
        self.assertIn("Jane Smith", names)
        self.assertIn("Bob Wilson", names)
        # "Dr. Sarah Chen" has more docs than "Sarah Chen", so it's canonical
        self.assertIn("Dr. Sarah Chen", names)

    def test_non_canonical_removed(self):
        """Non-canonical names should not exist in output entities."""
        conn = sqlite3.connect(self.output_db)
        c = conn.cursor()
        names = {row[0] for row in c.execute("SELECT name FROM entities").fetchall()}
        conn.close()

        self.assertNotIn("john doe", names)
        self.assertNotIn("JOHN DOE", names)
        self.assertNotIn("Mr. John Doe", names)
        self.assertNotIn("Acme Corp, Inc.", names)
        self.assertNotIn("acme corp", names)

    def test_documents_appeared_merged(self):
        """Canonical entity should have union of all members' documents."""
        conn = sqlite3.connect(self.output_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT documents_appeared FROM entities WHERE name = 'John Doe'").fetchone()
        conn.close()

        docs = set(json.loads(row["documents_appeared"]))
        # Union of doc1,doc2,doc3 + doc2 + doc4 + doc1 + doc5
        self.assertEqual(docs, {"doc1", "doc2", "doc3", "doc4", "doc5"})

    def test_relationships_rewritten(self):
        """Relationship source/target should use canonical names."""
        conn = sqlite3.connect(self.output_db)
        c = conn.cursor()

        # All "john doe" variants should now be "John Doe"
        rows = c.execute(
            "SELECT source, target FROM relationships WHERE source = 'john doe' OR target = 'john doe'"
        ).fetchall()
        self.assertEqual(len(rows), 0, "Non-canonical name 'john doe' still in relationships")

        rows = c.execute(
            "SELECT source, target FROM relationships WHERE source = 'Mr. John Doe' OR target = 'Mr. John Doe'"
        ).fetchall()
        self.assertEqual(len(rows), 0, "Non-canonical name 'Mr. John Doe' still in relationships")

        # "John Doe" should appear in relationships
        rows = c.execute(
            "SELECT * FROM relationships WHERE source = 'John Doe' OR target = 'John Doe'"
        ).fetchall()
        self.assertGreater(len(rows), 0)

        conn.close()

    def test_relationship_deduplication(self):
        """
        After rewriting, (John Doe, ACME Corp, works_for, doc1) appears from both
        original rel 1 (John Doe->ACME Corp) and rel 4 (Mr. John Doe->ACME Corp).
        These should be deduped to one.
        """
        conn = sqlite3.connect(self.output_db)
        c = conn.cursor()
        rows = c.execute(
            "SELECT * FROM relationships WHERE source = 'John Doe' AND target = 'ACME Corp' "
            "AND type = 'works_for' AND document_id = 'doc1'"
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1, "Duplicate relationship not deduped")

    def test_documents_table_updated(self):
        """Sender/recipient in documents table should use canonical names."""
        conn = sqlite3.connect(self.output_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM documents").fetchall()
        conn.close()

        senders = {row["sender"] for row in rows}
        recipients = {row["recipient"] for row in rows}

        # "john doe" (doc2 sender) should be rewritten to "John Doe"
        self.assertNotIn("john doe", senders)
        self.assertIn("John Doe", senders)

        # "Acme Corp, Inc." (doc3 sender) -> "ACME Corp"
        self.assertNotIn("Acme Corp, Inc.", senders)
        self.assertIn("ACME Corp", senders)

        # "U.S. EPA" is the canonical (same doc count, longer name wins tiebreak)
        # so it should remain as-is in the documents table
        self.assertIn("U.S. EPA", recipients)


class TestHTMLReport(unittest.TestCase):
    """Test that the HTML report generates without errors."""

    def test_report_generation(self):
        entities = [dict(zip(
            ["entity_id", "name", "type", "description", "first_seen_document", "documents_appeared"],
            row,
        )) for row in ENTITIES]

        groups, name_map = build_merge_groups(entities)

        html = generate_report(
            groups, name_map, "test_input.db", "test_output.db",
            len(entities), 10,
            {"final_entity_count": 6, "final_rel_count": 8,
             "relationships_rewritten": 5, "relationships_deduped": 2},
            0.1,
        )

        self.assertIn("Entity Normalization Report", html)
        self.assertIn("John Doe", html)
        self.assertIn("ACME Corp", html)
        self.assertIn("Merge Groups", html)
        # Should contain the JSON data blob
        self.assertIn("const GROUPS", html)


if __name__ == "__main__":
    unittest.main()
