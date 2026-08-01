# Mnemosyne

Persistent memory system for Large Language Models. Embed, extract, retrieve, remember.

Mnemosyne gives any LLM a persistent memory layer. Conversations are stored in SQLite, vector embeddings enable semantic search, and a knowledge graph captures structured facts — entities, relationships, and beliefs — across sessions.

## Install

```bash
pip install mnemosyne
```

Or with optional extras:

```bash
pip install mnemosyne[local]   # offline embeddings (sentence-transformers + spaCy)
pip install mnemosyne[vec]     # fast vector search (sqlite-vec)
pip install mnemosyne[faiss]   # fast vector search (FAISS)
```

## Quick Start

```python
from mnemosyne import MemoryEngine

# Create engine (reads GEMINI_API_KEY from env or .env file)
engine = MemoryEngine()

# Optional: eagerly load the embedding model
engine.warmup()

# Chat — stores message, extracts knowledge, retrieves context, calls LLM
result = engine.chat("I just started a new project called Atlas at work.")
print(result["response"])

# The memory system now knows about "Atlas" and your work context
result = engine.chat("What project am I working on?")
print(result["response"])  # -> "You mentioned starting a new project called Atlas at work."
```

## How It Works

Every call to `engine.chat()` runs a full pipeline:

1. **Store** — your message is saved to SQLite
2. **Embed** — a vector representation is computed and stored
3. **Retrieve** — a deterministic pipeline finds relevant knowledge:
   - Query planning (classify intent, detect entities)
   - Entity resolution (exact, substring, fuzzy matching)
   - Graph traversal (BFS over the knowledge graph)
   - Vector search (cosine similarity over embeddings)
   - Ranking (6-signal weighted scoring)
   - Deduplication, compression, context assembly
4. **Generate** — the LLM responds with structured memory context
5. **Extract** — entities, relationships, and facts are pulled from the exchange
6. **Store ontology** — new knowledge is added to the graph and database

## Configuration

Set via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Your Gemini API key |
| `GEMINI_LLM_MODEL` | `gemini-2.0-flash` | Chat model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `EMBEDDING_BACKEND` | `local` | `local` (sentence-transformers) or `gemini` (API) |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `DATABASE_URL` | `sqlite:///mnemosyne.db` | Database path |
| `VEC_TOP_K` | `10` | Default vector search results |

## API Reference

### `MemoryEngine`

The main orchestrator. Coordinates storage, embedding, retrieval, LLM calls, and knowledge extraction.

```python
engine = MemoryEngine(
    embeddings=EmbeddingService(),  # optional custom embedding backend
    graph=GraphService(),           # optional custom graph backend
    llm=LLMService(),              # optional custom LLM backend
    retrieval=RetrievalService(),   # optional custom retrieval pipeline
)
```

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `chat(message, session_id=None)` | `dict` | Process message through full pipeline. Returns `{"response", "pipeline", "session_id"}` |
| `search_memory(query, top_k=10)` | `list[dict]` | Vector similarity search over stored messages |
| `search_graph(entity_name)` | `dict` | Search knowledge graph for entity and its neighborhood |
| `get_neighbors(entity, hops=1)` | `dict` | BFS traversal from entity in knowledge graph |
| `warmup()` | `None` | Eagerly load the embedding model |
| `reload_config()` | `None` | Reload configuration from database |
| `is_ready` | `bool` | Whether the embedding model is loaded |

### `EmbeddingService`

Vector embedding and search.

```python
from mnemosyne import EmbeddingService

emb = EmbeddingService()
emb.warmup()

vector = emb.embed("hello world")           # -> list[float]
vectors = emb.embed_batch(["a", "b"])       # -> list[list[float]]
results = emb.search(db, "hello", top_k=5)  # -> list[dict]
```

### `GraphService`

NetworkX-based knowledge graph.

```python
from mnemosyne import GraphService

graph = GraphService()
graph.load_from_db(db)

graph.add_entity("Alice", "PERSON", confidence=0.9)
graph.add_relationship("Alice", "works_for", "Acme Corp", confidence=0.8)

neighbors = graph.get_neighbors("Alice", hops=2)
search = graph.search_entity("Ali")  # substring match
```

### `LLMService`

LLM provider abstraction (Gemini, OpenAI, DeepSeek, Grok, Kimi, Ollama).

```python
from mnemosyne import LLMService

llm = LLMService()
response = llm.chat([{"role": "user", "content": "Hello"}])
structured = llm.chat_json([{"role": "user", "content": "Extract entities"}])
```

### `ContextPipeline`

Deterministic retrieval pipeline (no LLM calls).

```python
from mnemosyne import ContextPipeline

pipeline = ContextPipeline(embeddings, graph, max_hops=3, char_budget=8000)
result = pipeline.run(db, query, conversation, query_vector=vector)
# result.plan, result.resolved_entities, result.graph_result, result.memory_result, result.context
```

## Database

Mnemosyne uses SQLite with WAL mode. Seven tables:

| Table | Purpose |
|---|---|
| `chat_sessions` | Conversation sessions |
| `messages` | Messages (role, content, timestamp) |
| `embeddings` | Vector embeddings (binary-packed float32) |
| `entities` | Named entities (name, type, confidence) |
| `relationships` | Subject-predicate-object triples |
| `facts` | Atomic facts with source messages |
| `settings` | Runtime configuration |

## Development

```bash
git clone https://github.com/elitsecorp/mnemosyne.git
cd mnemosyne
uv sync --extra dev
uv run pytest tests/ -v
```

## License

MIT
