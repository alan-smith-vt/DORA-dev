"""
extract_entities.py - Extract entities and relationships from correspondence using local LLM.

Simple script to process all document chunks and extract structured JSON for graph RAG.
Just run:
    python extract_entities.py

Requires:
    - Ollama running locally (ollama serve)
    - Qwen2.5 14B model pulled (ollama pull qwen2.5:14b)
    - Existing chunk database from document ingestion

Configuration is hardcoded below for simplicity.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm
from datetime import datetime


# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================

CHUNK_DB = "BAYT_PROD_chunks.db"
ENTITY_DB = "BAYT_PROD_entities.db"
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# System prompt for entity extraction from engineering/legal correspondence
SYSTEM_PROMPT = """You are an expert at extracting entities and relationships from engineering and legal correspondence.

    Extract the following from the letter:
    1. METADATA: Sender, recipient, date, subject, reference numbers
    2. ENTITIES: People, organizations, projects, locations, issues/events, documents, equipment
    3. RELATIONSHIPS: How entities relate to each other in this correspondence

    Return your response as valid JSON with this exact structure:
    {
      "metadata": {
        "sender": "organization or person name",
        "recipient": "organization or person name",
        "cc": "organization or person name (or null)",
        "bcc": "organization or person name (or null)",
        "date": "date if mentioned (YYYY-MM-DD format or null)",
        "subject": "subject line or topic",
        "reference": "reference number or project code if mentioned"
      },
      "entities": [
        {
          "name": "entity name",
          "type": "person|organization|project|location|document|equipment|regulation|issue",
          "description": "brief description or role"
        }
      ],
      "relationships": [
        {
          "source": "entity name",
          "target": "entity name",
          "type": "works_for|responsible_for|inspected|documented|supplied|violated|noted|disputed|caused|remediated",
          "description": "brief description of relationship from this letter"
        }
      ]
    }

    IMPORTANT:
    - Only extract information explicitly mentioned in the text
    - Use consistent entity names (e.g., always "John Doe" not variations)
    - Issues/events (deficiencies, violations, incidents, inspections) should be entities
    - Use relationships to connect people/orgs to issues (e.g., "John Doe" -[noted]-> "Phase 3 Foundation Cracking")
    - Relationship source and target must match entity names exactly
    - For sender/recipient/cc/bcc, extract the organization or person name, not just titles
    - Keep descriptions concise (one sentence)
    - Return ONLY valid JSON, no other text"""


# ============================================================================
# Database Setup
# ============================================================================

def create_entity_database(db_path: str):
    """Create SQLite database for storing extracted entities and relationships."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Documents metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
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
    
    # Entities table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
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
    
    # Relationships table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            document_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Processing log table (track which documents have been processed)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_log (
            document_id TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN,
            error_message TEXT
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationship_source ON relationships(source)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationship_target ON relationships(target)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationship_document ON relationships(document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_sender ON documents(sender)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_recipient ON documents(recipient)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_reference ON documents(reference)')
    
    conn.commit()
    conn.close()


# ============================================================================
# LLM Interaction
# ============================================================================

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call local Ollama instance with a prompt.
    
    Args:
        prompt: The prompt to send
        model: Model name to use
        
    Returns:
        Model response as string
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent extraction
            "num_ctx": 8192      # Context window
        }
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result.get("response", "")


def extract_entities_from_document(document_text: str) -> Dict:
    """
    Extract entities, relationships, and metadata from a document using LLM.
    
    Args:
        document_text: The document text to process
        
    Returns:
        Dictionary with 'metadata', 'entities', 'relationships' keys
    """
    # Construct the full prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nLetter text:\n{document_text}\n\nJSON output:"
    #print("*"*20)
    #print(full_prompt)
    #print("*"*20)
    
    # Call LLM
    response = call_ollama(full_prompt)
    
    # Parse JSON from response
    # Sometimes models include markdown code blocks, so we need to strip those
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    
    # Parse JSON
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        print(f"Response was: {response[:200]}...")
        return {
            "metadata": {},
            "entities": [],
            "relationships": [],
        }


# ============================================================================
# Database Storage
# ============================================================================

def store_document_metadata(conn: sqlite3.Connection, metadata: Dict, document_id: str, source_file: str):
    """Store document metadata."""
    cursor = conn.cursor()
    
    # Helper to handle potential lists
    def normalize_field(value):
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v)
        return str(value)
    
    cursor.execute('''
        INSERT OR REPLACE INTO documents 
        (document_id, sender, recipient, cc, bcc, date, subject, reference, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        document_id,
        normalize_field(metadata.get("sender")),
        normalize_field(metadata.get("recipient")),
        normalize_field(metadata.get("cc")),
        normalize_field(metadata.get("bcc")),
        normalize_field(metadata.get("date")),
        normalize_field(metadata.get("subject")),
        normalize_field(metadata.get("reference")),
        source_file
    ))

def store_entities(conn: sqlite3.Connection, entities: List[Dict], document_id: str):
    """Store extracted entities in the database."""
    cursor = conn.cursor()
    
    for entity in entities:
        name = entity.get("name")
        entity_type = entity.get("type")
        description = entity.get("description")
        
        # Skip if name or type is None/empty
        if not name or not entity_type:
            continue
        
        # Now safe to strip
        name = name.strip()
        entity_type = entity_type.strip()
        description = description.strip() if description else ""
        
        # Check if entity already exists
        cursor.execute('''
            SELECT entity_id, documents_appeared FROM entities 
            WHERE name = ? AND type = ?
        ''', (name, entity_type))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing entity
            entity_id, documents_appeared = result
            docs_list = json.loads(documents_appeared) if documents_appeared else []
            if document_id not in docs_list:
                docs_list.append(document_id)
            
            cursor.execute('''
                UPDATE entities 
                SET documents_appeared = ?
                WHERE entity_id = ?
            ''', (json.dumps(docs_list), entity_id))
        else:
            # Insert new entity
            cursor.execute('''
                INSERT INTO entities (name, type, description, first_seen_document, documents_appeared)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, entity_type, description, document_id, json.dumps([document_id])))


def store_relationships(conn: sqlite3.Connection, relationships: List[Dict], document_id: str):
    """Store extracted relationships in the database."""
    cursor = conn.cursor()
    
    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        rel_type = rel.get("type")
        description = rel.get("description")
        
        # Skip if source, target, or type is None/empty
        if not source or not target or not rel_type:
            continue
        
        # Now safe to strip
        source = source.strip()
        target = target.strip()
        rel_type = rel_type.strip()
        description = description.strip() if description else ""
        
        cursor.execute('''
            INSERT INTO relationships (source, target, type, description, document_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (source, target, rel_type, description, document_id))


def log_processing(conn: sqlite3.Connection, document_id: str, success: bool, error_message: str = None):
    """Log that a document has been processed."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processing_log (document_id, success, error_message)
        VALUES (?, ?, ?)
    ''', (document_id, success, error_message))


# ============================================================================
# Chunk Database Interface
# ============================================================================

class ChunkDatabase:
    """Simple interface to the chunk database."""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def get_all_chunks(self) -> List[Dict]:
        """Get all chunks from database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM chunks ORDER BY chunk_id')
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================================
# Main Processing
# ============================================================================

# def main():
print("Engineering Correspondence - Entity Extraction")
print("=" * 80)

# Check if chunk database exists
if not Path(CHUNK_DB).exists():
    print(f"ERROR: Chunk database '{CHUNK_DB}' not found!")
    print("Please run document ingestion script first.")
    # return 1

# Check if Ollama is running
try:
    test_response = requests.get("http://localhost:11434/api/tags")
    test_response.raise_for_status()
except requests.exceptions.RequestException:
    print("ERROR: Cannot connect to Ollama!")
    print("Please ensure Ollama is running: ollama serve")
    # return 1

# Check if model is available
try:
    models = requests.get("http://localhost:11434/api/tags").json()
    model_names = [m["name"] for m in models.get("models", [])]
    if OLLAMA_MODEL not in model_names:
        print(f"ERROR: Model '{OLLAMA_MODEL}' not found!")
        print(f"Please pull the model: ollama pull {OLLAMA_MODEL}")
        # return 1
except Exception as e:
    print(f"Warning: Could not verify model availability: {e}")

# Create entity database
print(f"\nInitializing entity database: {ENTITY_DB}")
create_entity_database(ENTITY_DB)

# Open both databases
chunk_db = ChunkDatabase(CHUNK_DB)
entity_conn = sqlite3.connect(ENTITY_DB)

# Get all documents (chunks in this case are full documents)
all_documents = chunk_db.get_all_chunks()

# Check which documents are already processed
entity_cursor = entity_conn.cursor()
entity_cursor.execute('SELECT document_id FROM processing_log WHERE success = 1')
processed_doc_ids = set(row[0] for row in entity_cursor.fetchall())

# Filter to only unprocessed documents
docs_to_process = [doc for doc in all_documents if str(doc['chunk_id']) not in processed_doc_ids]

print(f"\nTotal documents in database: {len(all_documents)}")
print(f"Already processed: {len(processed_doc_ids)}")
print(f"Remaining to process: {len(docs_to_process)}")

if len(docs_to_process) == 0:
    print("\nAll documents already processed!")
    chunk_db.close()
    entity_conn.close()
    # return 0

print(f"Using model: {OLLAMA_MODEL}")
print("-" * 80)

# Process each unprocessed document
total_docs = len(docs_to_process)
processed_count = 0
error_count = 0

for i, doc in enumerate(tqdm(docs_to_process, desc="Processing documents"), 1):
    document_id = doc['chunk_id']
    text = doc['text']
    source_file = doc.get('source_file', '')
    
    try:
        # Extract entities, relationships, and metadata
        result = extract_entities_from_document(text)
        
        metadata = result.get("metadata", {})
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        # print(metadata)
        # print(entities)
        # print(relationships)
        
        # Store in database
        store_document_metadata(entity_conn, metadata, document_id, source_file)
        store_entities(entity_conn, entities, document_id)
        store_relationships(entity_conn, relationships, document_id)
        log_processing(entity_conn, document_id, True)
        
        # Commit after each document
        entity_conn.commit()
        
        processed_count += 1
        
    except Exception as e:
        print(f"\nERROR processing {document_id}: {e}")
        log_processing(entity_conn, document_id, False, str(e))
        entity_conn.commit()
        error_count += 1

# Final statistics
print("\n" + "=" * 80)
print("Processing Complete!")
print("=" * 80)
print(f"Successfully processed: {processed_count}/{total_docs} documents")
print(f"Errors: {error_count}")

# Get extraction statistics
cursor = entity_conn.cursor()
cursor.execute("SELECT COUNT(*) FROM entities")
total_entities = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM relationships")
total_relationships = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(DISTINCT sender) FROM documents WHERE sender != ''")
unique_senders = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(DISTINCT recipient) FROM documents WHERE recipient != ''")
unique_recipients = cursor.fetchone()[0]

print(f"\nExtracted:")
print(f"  - {total_entities} unique entities")
print(f"  - {total_relationships} relationships")
print(f"  - {unique_senders} unique senders")
print(f"  - {unique_recipients} unique recipients")
print(f"\nDatabase saved to: {ENTITY_DB}")

# Close databases
chunk_db.close()
entity_conn.close()
    
    # return 0


# if __name__ == "__main__":
#     exit(main())