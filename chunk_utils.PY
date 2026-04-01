import os
import sqlite3
from pathlib import Path
from typing import Dict

class ChunkDatabase:
	"""SQLite database manager for storing and retrieving text chunks."""
	
	def __init__(self, db_path: str = "chunks.db", overwrite: bool = False):
		"""
		Initialize SQLite database connection.
		
		Args:
			db_path: Path to SQLite database file
			overwrite: If True, delete existing database without confirmation
		"""
		self.db_path = db_path
		
		# Check if database exists and handle accordingly
		if os.path.exists(db_path):
			if not overwrite:
				print(f"\n⚠️  WARNING: Database '{db_path}' already exists!")
				response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
				if response not in ['yes', 'y']:
					raise FileExistsError(f"Database '{db_path}' already exists. Operation cancelled.")
			
			# Remove existing database
			print(f"Deleting existing database: {db_path}")
			os.remove(db_path)
		
		self.conn = sqlite3.connect(db_path)
		self.cursor = self.conn.cursor()
		self._create_tables()
	
	def _create_tables(self):
		"""Create the chunks table if it doesn't exist."""
		self.cursor.execute('''
			CREATE TABLE IF NOT EXISTS chunks (
				chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
				filename TEXT NOT NULL,
				text TEXT NOT NULL,
				token_count INTEGER,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
		''')
		
		# Create index for efficient filename queries
		self.cursor.execute('''
			CREATE INDEX IF NOT EXISTS idx_filename 
			ON chunks(filename)
		''')
		
		self.conn.commit()
	
	def insert_chunk(self, filename: str, text: str, token_count: int):
		"""
		Insert a single chunk into the database.
		
		Args:
			filename: Name of the source file
			text: The actual text content
			token_count: Estimated token count
		"""
		self.cursor.execute('''
			INSERT INTO chunks 
			(filename, text, token_count)
			VALUES (?, ?, ?)
		''', (filename, text, token_count))
	
	def commit(self):
		"""Commit the current transaction."""
		self.conn.commit()
	
	def get_stats(self) -> Dict:
		"""
		Get statistics about the chunks in the database.
		
		Returns:
			Dictionary containing database statistics
		"""
		self.cursor.execute('SELECT COUNT(*) FROM chunks')
		total_chunks = self.cursor.fetchone()[0]
		
		self.cursor.execute('SELECT COUNT(DISTINCT filename) FROM chunks')
		total_files = self.cursor.fetchone()[0]
		
		self.cursor.execute('SELECT AVG(token_count) FROM chunks')
		avg_tokens = self.cursor.fetchone()[0]
		
		self.cursor.execute('SELECT SUM(token_count) FROM chunks')
		total_tokens = self.cursor.fetchone()[0]
		
		return {
			'total_chunks': total_chunks,
			'total_files': total_files,
			'avg_tokens_per_chunk': round(avg_tokens, 2) if avg_tokens else 0,
			'total_tokens': total_tokens if total_tokens else 0
		}
	
	def close(self):
		"""Close the database connection."""
		self.conn.close()
	
	def __enter__(self):
		"""Context manager entry."""
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Context manager exit - automatically commits and closes."""
		if exc_type is None:
			self.commit()
		self.close()