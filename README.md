# Mnemosyne

A persistent memory system for Large Language Models. Mnemosyne stores both **semantic** (vector embeddings) and **structured** (knowledge graph) representations of knowledge, enabling LLMs to remember conversations, entities, and relationships across sessions.

## What It Does

Mnemosyne sits between the user and any OpenAI-compatible LLM. Every conversation is:

1. **Stored** — messages persist in SQLite
2. **Embedded** — vector representations enable semantic search
3. **Extracted** — entities, relationships, and facts are pulled into a knowledge graph
4. **Retrieved** — a deterministic pipeline finds relevant knowledge for each query
5. **Presented** — structured context is sent to the LLM for language generation

The LLM generates language. Mnemosyne handles memory.

## Architecture

```
User message
    │
    ├──► Query Planner (deterministic)
    ├──► Entity Resolver (exact, substring, fuzzy match)
    ├──► Graph Retriever (BFS traversal, ranked)
    ├──► Memory Retriever (vector search, conditional)
    ├──► Ranker (6-signal weighted scoring)
    ├──► Deduplicator
    ├──► Compressor (token budget)
    ├──► Context Builder (structured sections)
    │
    └──► LLM (language generation only)
```

### Key Design Decisions

- **Deterministic context generation** — Mnemosyne performs retrieval, ranking, filtering, and context construction. The LLM only generates language.
- **Dual memory representations** — Vector embeddings for semantic similarity, knowledge graph for structured facts.
- **Conditional vector search** — Vector search runs for all queries but with quality filters (min_similarity=0.6).
- **Explainable pipeline** — every response includes a "Show pipeline details" dropdown showing exactly what was retrieved and why.
- **Owner Graph** — automatically connects concepts to the owner entity for personalized retrieval.
- **Offline embeddings** — uses sentence-transformers (all-MiniLM-L6-v2) for zero-cost vector search.
- **OpenIE extraction** — rule-based triple extraction for document ingestion (zero cost).

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Backend | FastAPI |
| Database | SQLite (WAL mode) |
| ORM | SQLAlchemy 2.x |
| Vector Search | sqlite-vec / FAISS / numpy brute-force |
| Graph | NetworkX |
| LLM | Google Gemini (configurable) |
| Embeddings | sentence-transformers (local) or Gemini |
| OpenIE | Rule-based regex extraction |
| Validation | Pydantic |
| Package Manager | uv |

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Steps

```bash
# Clone the repository
git clone https://github.com/elitsecorp/mnimosyne.git
cd mnemosyne

# Install dependencies (includes local embeddings + spaCy)
uv sync --extra local

# Install dev dependencies (for running tests)
uv sync --extra dev

# Configure environment
cp .env.example .env
```

Edit `.env` with your settings:

```
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_LLM_MODEL=gemini-3.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_BACKEND=local
```

### Run

```bash
uv run python main.py
```

Server starts at `http://localhost:8000`.

### Run Tests

```bash
uv run python -m pytest tests/ -v
```

### Run Evaluation

```bash
uv run python -m mnemosyne.evaluation.run_eval
```

## Usage

### Chat

Open `http://localhost:8000` in your browser. Type a message and Mnemosyne will:

1. Store your message
2. Retrieve relevant memories from the knowledge graph and vector store
3. Build structured context
4. Send it to the LLM
5. Extract new entities, relationships, and facts from the response
6. Store them in the graph for future queries

Every response includes a **"Show pipeline details"** button that reveals exactly what was retrieved, ranked, and sent to the LLM.

### Chat Sessions

- Create new sessions with **+ New Chat**
- Switch between sessions by clicking on them
- Delete sessions with the **x** button
- All conversations are stored in the memory system regardless of session

### Memory Explorer

Open `http://localhost:8000/memories` to browse all stored data:

- **Entities** — named things (people, places, organizations)
- **Relationships** — how entities relate (subject → predicate → object)
- **Facts** — atomic statements with source messages
- **Messages** — full conversation history

### Knowledge Graph

Open `http://localhost:8000/graph-explorer` for an interactive graph visualization powered by Cytoscape.js:

- Zoom, pan, drag nodes
- Click nodes to see details and connected relationships
- Double-click to expand neighbors (lazy loading)
- Search entities by name
- Filter by entity type, confidence, and date
- **Owner** button to view the owner-centric subgraph

### Data Import

Open `http://localhost:8000/importer` to import external text:

- Paste text, markdown, or conversation transcripts
- Upload .txt or .md files
- Text is chunked and embedded (zero cost with local embeddings)
- Progress bar shows chunks processed, entities, relationships, embeddings

### Memory Consolidator

Open `http://localhost:8000/consolidate` to improve graph quality:

- **Duplicate entity detection** — find and merge similar entities
- **Relationship normalization** — merge synonymous predicates
- **Orphan detection** — find entities with no relationships
- **Unsupported relationships** — find relationships without evidence
- **Confidence recalculation** — update scores based on evidence
- **Owner connections** — automatically link concepts to the owner

All changes are advisory — review and approve before applying. Auto-consolidation runs every 5 prompts.

### Reset Database

Click **Reset DB** in the nav bar on any page to delete all data and start fresh.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `GET` | `/memories` | Memory explorer |
| `GET` | `/graph-explorer` | Knowledge graph |
| `GET` | `/importer` | Data import |
| `GET` | `/consolidate` | Memory consolidation |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send a message |
| `POST` | `/search` | Vector similarity search |
| `POST` | `/api/database/reset` | Delete all data |
| `POST` | `/api/sessions` | Create session |
| `GET` | `/api/sessions` | List sessions |
| `DELETE` | `/api/sessions/{id}` | Delete session |
| `GET` | `/api/entities` | List all entities |
| `GET` | `/api/relationships` | List all relationships |
| `GET` | `/api/facts` | List all facts |
| `GET` | `/api/messages` | List all messages |
| `GET` | `/api/stats` | Summary counts |
| `GET` | `/api/graph/seed` | Graph seed for Cytoscape |
| `GET` | `/api/graph/search?q=` | Search entities |
| `GET` | `/api/graph/node/{id}` | Node detail |
| `GET` | `/api/graph/neighbors/{id}` | Lazy expansion |
| `GET` | `/api/owner/graph` | Owner subgraph |
| `POST` | `/api/import/text` | Import text |
| `POST` | `/api/import/file` | Import file |
| `GET` | `/api/import/job/{id}` | Import progress |
| `POST` | `/api/consolidate/analyze` | Run consolidation |
| `POST` | `/api/consolidate/apply` | Apply recommendations |

## Project Structure

```
mnemosyne/
├── main.py                    # Entry point
├── pyproject.toml             # Project config
├── .env.example               # Environment template
├── mnemosyne/
│   ├── __init__.py
│   ├── config.py              # Settings from environment
│   ├── database.py            # SQLAlchemy engine + sessions
│   ├── models.py              # ORM models (6 tables)
│   ├── schemas.py             # Pydantic validation
│   ├── llm.py                 # Gemini LLM client
│   ├── embeddings.py          # Embedding service + vector search
│   ├── graph.py               # NetworkX knowledge graph
│   ├── extraction.py          # Knowledge extraction from LLM
│   ├── prompts.py             # Prompt templates
│   ├── memory.py              # Memory engine (orchestrator)
│   ├── api.py                 # FastAPI routes
│   ├── static.html            # Chat UI
│   ├── memories.html          # Memory explorer UI
│   ├── graph_explorer.html    # Knowledge graph UI
│   ├── importer.html          # Data import UI
│   ├── consolidator.html      # Memory consolidation UI
│   ├── static/
│   │   └── cytoscape.min.js   # Cytoscape.js library
│   ├── retrieval/             # Deterministic context pipeline
│   │   ├── planner.py         # Query analysis
│   │   ├── resolver.py        # Entity resolution
│   │   ├── graph_retriever.py # Graph traversal
│   │   ├── memory_retriever.py# Vector search
│   │   ├── ranker.py          # Multi-signal scoring
│   │   ├── deduplicator.py    # Deduplication
│   │   ├── compressor.py      # Token budget
│   │   ├── builder.py         # Context construction
│   │   ├── pipeline.py        # Pipeline orchestrator
│   │   ├── summarizer.py      # Offline text summarization
│   │   └── wrapper.py         # Backward-compatible interface
│   ├── services/
│   │   ├── graph_explorer.py  # Graph exploration
│   │   ├── importer.py        # Text ingestion
│   │   ├── consolidation.py   # Memory consolidation
│   │   ├── owner_compiler.py  # Owner graph compilation
│   │   └── openie.py          # Rule-based triple extraction
│   └── evaluation/
│       ├── harness.py         # Evaluation framework
│       ├── run_eval.py        # Evaluation runner
│       └── test_cases.json    # Test cases
└── tests/
    ├── test_memory.py
    ├── test_consolidation.py
    ├── test_importer.py
    ├── test_pipeline.py
    ├── test_owner.py
    └── test_openie.py
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Your Gemini API key |
| `GEMINI_LLM_MODEL` | `gemini-3.5-flash` | Chat model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `EMBEDDING_BACKEND` | `local` | `local` or `gemini` |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `DATABASE_URL` | `sqlite:///mnemosyne.db` | Database path |
| `VEC_TOP_K` | `10` | Default vector search results |

## Database Schema

| Table | Description |
|---|---|
| `chat_sessions` | Chat sessions (title, timestamps) |
| `messages` | Conversation messages (role, content, session, timestamp) |
| `embeddings` | Vector embeddings linked to messages |
| `entities` | Named entities (name, type, confidence) |
| `relationships` | Entity relationships (subject, predicate, object, is_owner) |
| `facts` | Atomic facts with source messages |

## License

MIT
