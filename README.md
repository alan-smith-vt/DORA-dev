# DORA

**Document Organization and Relationship Analysis**

DORA is a tool for creating knowledge graphs from document collections in a fully secure offline environment. Using locally-run large language models, DORA parses any number of documents based on user-defined areas of interest, extracting entities, relationships, and metadata to build an interactive knowledge graph.

## Features

- **Fully Offline**: All processing happens locally with no external API calls
- **Secure**: Documents never leave your machine
- **Customizable**: System prompts and entity extraction can be tailored to specific use cases
- **Scalable**: Optimized for parallel processing on capable hardware
- **Interactive Visualization**: Generates standalone HTML knowledge graphs with filtering and search capabilities

## Core Components

### create_chunk_database.py

The primary script for parsing document folder hierarchies and creating a text chunk database. This script:

- Traverses the document directory structure
- Extracts text from PDFs on a page-by-page basis
- Creates a database with metadata including filepath and page numbers
- Relies on `chunk_utils.py` for database operations

**Current Limitations**: 
- Requires modification for different folder locations
- Currently supports PDFs with extractable text only (non-image-based PDFs)

**Future Enhancements**: 
- Support for additional file types (DOCX, TXT, etc.)
- Improved robustness for various document formats

### chunk_utils.py

Handles all database operations for the chunk database, providing utilities for creating, reading, and managing the text chunk storage layer.

### extract_entities.py

The intelligence core of DORA. This script:

- Guides the LLM through reading each document page
- Extracts entities, relationships, and metadata based on system prompt instructions
- Returns structured JSON output conforming to the defined schema
- Creates the entities database that drives the visualization layer

**System Prompt**: The system prompt contains the core logic for LLM instructions. Currently embedded in the script, it should ideally be externalized for per-run customization.

**Performance Optimization**: Can be configured to work with `OLLAMA_NUM_PARALLEL > 1` for significant performance gains on systems with sufficient GPU memory. For example, with a 24GB GPU running a 12GB model, setting `num_parallel` to 5-10 can yield up to 6x performance improvements by preloading data and minimizing downtime between queries (this does not load multiple model copies, only optimizes data throughput).

### visualize_graph.py

Generates the interactive knowledge graph visualization using pyvis. This script:

- Reads the entities database
- Organizes entities and relationships into a network graph
- Outputs a standalone HTML file for instant viewing
- Includes built-in UI controls and filtering tools

**Note**: JavaScript code for the UI and filtering is currently embedded as raw text within the Python script and could be externalized for easier maintenance.

## Requirements

- Python 3.x
- Local LLM runtime (e.g., Ollama)
- Sufficient GPU memory for model inference
- Required Python packages (see `requirements.txt`)

## Workflow

1. **Document Preparation**: Organize documents in a folder structure
2. **Chunk Creation**: Run `create_chunk_database.py` to parse and store text chunks
3. **Entity Extraction**: Execute `extract_entities.py` to identify entities and relationships
4. **Visualization**: Generate the knowledge graph with `visualize_graph.py`
5. **Exploration**: Open the resulting HTML file in any web browser

## Database Schema

The system uses two primary databases:

- **Chunk Database**: Stores text chunks with metadata (filepath, page number, etc.)
- **Entity Database**: Stores extracted entities, relationships, and associated metadata

The structure of these databases is interdependent, with the entity extraction phase relying on metadata from the chunk database to enable features like source linking and page number references in the visualization.

## Future Development

- Externalize system prompts for easier customization
- Expand file type support beyond extractable-text PDFs
- Separate JavaScript UI code from Python visualization script
- Add configuration file support for folder paths and processing parameters
- Implement automatic optimization for parallel processing based on available GPU memory

## Use Cases

- Legal document analysis and evidence tracking
- Research paper relationship mapping
- Corporate document organization
- Forensic analysis of document collections
- Knowledge base creation from unstructured documents

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]