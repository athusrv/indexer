# context-db

A **production-quality filesystem indexing engine** for AI agents.

Converts a local directory — source code, PDFs, spreadsheets, Word documents, or any mix — into a persistent, searchable SQLite database so language models can retrieve relevant context without re-exploring the repository on every request.

Supports three retrieval strategies out of the box:

```
ctx search "jwt"                       # fast BM25 lexical search (default)
ctx search "token validation" --semantic   # dense vector cosine search
ctx search "error handling"  --hybrid      # weighted FTS + semantic rerank
```

---

## Architecture

### Full system overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLI  (Typer)                               │
│   ctx init │ ctx index │ ctx search │ ctx stats │ ctx reset          │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │      Indexing Pipeline     │
         │  scan → diff → extract     │
         │  → chunk → persist         │
         │  → embed (optional)        │
         └──────┬───────────┬─────────┘
                │           │
   ┌────────────▼──┐  ┌─────▼──────────────┐
   │    Scanner    │  │   Fingerprinter     │
   │  os.walk +    │  │  SHA-256 hashing    │
   │  ignore rules │  │  change detection   │
   └───────┬───────┘  └────────────────────┘
           │
   ┌───────▼──────────────┐
   │      Extractor       │
   │  plain text: UTF-8   │
   │  rich: markitdown    │
   │  (PDF/DOCX/XLSX/…)   │
   └───────┬──────────────┘
           │
   ┌───────▼──────────────┐
   │       Chunker        │
   │  line-aligned slices │
   │  configurable overlap│
   └───────┬──────────────┘
           │
   ┌───────▼──────────────────────────────────────┐
   │                Storage Layer                 │
   │                                              │
   │  SQLite  ┌──────────┐  ┌───────────────────┐│
   │          │  files   │  │      chunks       ││
   │          └──────────┘  └────┬──────────────┘│
   │                             │ trigger        │
   │                        ┌────▼──────┐         │
   │                        │chunks_fts │         │
   │                        │  (FTS5)   │         │
   │                        └───────────┘         │
   │                        ┌───────────────────┐ │
   │                        │ chunk_embeddings  │ │
   │                        │  (float32 blobs)  │ │
   │                        └───────────────────┘ │
   └──────────────────────────────┬───────────────┘
                                  │
   ┌──────────────────────────────▼──────────────────────┐
   │                   Retrieval Layer                   │
   │                                                     │
   │  ┌────────────┐  ┌──────────────┐  ┌─────────────┐ │
   │  │ SearchEngine│  │semantic_search│  │hybrid_search│ │
   │  │ FTS5 + BM25│  │cosine top-k  │  │0.6·lex +    │ │
   │  │ dedup      │  │brute-force   │  │0.4·sem      │ │
   │  └────────────┘  └──────────────┘  └─────────────┘ │
   └─────────────────────────────────────────────────────┘
```

### Hybrid search flow

```
query
  │
  ├─► FTS5 Search  (top-50 chunks by BM25)
  │
  └─► Embedding Search  (top-50 chunks by cosine similarity)
  │
  ▼
Merge  (union on chunk_id, determine match_type)
  │
  ▼
Normalise  (min-max each score set independently to [0, 1])
  │
  ▼
Rerank  final = 0.60 × lexical_norm + 0.40 × semantic_norm
  │
  ▼
list[HybridResult(path, score, match_type, preview)]
```

`match_type` is `"lexical"` | `"semantic"` | `"hybrid"` depending on which signal(s) found the chunk.

### Indexing data flow

```
Filesystem
    │
    ▼ Scanner       → [DiscoveredFile, ...]          (skips binaries, ignored dirs)
    │
    ▼ Fingerprinter → ChangeSet (new / modified / deleted)
    │
    ▼ Extractor     → plain text or Markdown         (UTF-8 / latin-1 / markitdown)
    │
    ▼ Chunker       → [Chunk(path, start_line, end_line, content), ...]
    │
    ▼ Repository    → SQLite files + chunks + FTS5   (transactional upsert)
    │
    ▼ Embedder      → chunk_embeddings               (optional; skips unchanged chunks)
```

### SQLite schema

```sql
-- file registry
CREATE TABLE files (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    path  TEXT    UNIQUE NOT NULL,
    hash  TEXT    NOT NULL,   -- hex SHA-256
    mtime REAL    NOT NULL
);

-- content chunks
CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    content    TEXT    NOT NULL
);

-- full-text search index (BM25 via FTS5)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- dense vector embeddings (optional — populated by --embed-model)
CREATE TABLE chunk_embeddings (
    chunk_id   INTEGER PRIMARY KEY,
    embedding  BLOB    NOT NULL,   -- float32 little-endian array
    model      TEXT    NOT NULL,   -- e.g. "nomic-ai/nomic-embed-text-v1.5"
    dimensions INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);
```

Triggers keep `chunks_fts` in sync with `chunks` automatically (INSERT / UPDATE / DELETE).  
`chunk_embeddings` cascades on chunk deletion — modified file re-index always produces fresh vectors.

---

## Project layout

```
context-db/
├── context_db/
│   ├── models.py                # Pydantic data models (shared across all layers)
│   ├── cli.py                   # Typer CLI entry-point
│   ├── indexer/
│   │   ├── scanner.py           # Filesystem traversal + ignore rules
│   │   ├── fingerprint.py       # SHA-256 hashing + change detection
│   │   ├── extractor.py         # Rich-format text extraction (markitdown)
│   │   ├── chunker.py           # Line-aligned character chunking
│   │   └── pipeline.py          # Orchestration: scan→diff→chunk→persist→embed
│   ├── embeddings/
│   │   ├── embedder.py          # Embedder ABC + create_embedder() factory
│   │   └── providers/
│   │       └── local.py         # LocalEmbedder (sentence-transformers, lazy load)
│   ├── storage/
│   │   ├── db.py                # SQLite connection + numbered migrations
│   │   └── repository.py        # Repository pattern (all SQL here)
│   └── retrieval/
│       ├── search.py            # FTS5 + BM25 search, file-level deduplication
│       ├── semantic.py          # Cosine similarity search (brute-force, numpy/pure-py)
│       └── hybrid.py            # Weighted FTS + semantic merge and rerank
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_scanner.py
│   ├── test_fingerprint.py
│   ├── test_chunker.py
│   ├── test_extractor.py
│   ├── test_storage.py
│   ├── test_pipeline.py
│   ├── test_search.py
│   ├── test_cli.py
│   ├── test_edge_cases.py
│   ├── test_embedding_storage.py    # migration 2, CRUD, cascade delete
│   ├── test_embedder.py             # Embedder ABC + LocalEmbedder (mocked ST)
│   ├── test_semantic.py             # cosine search, numpy + pure-python paths
│   ├── test_hybrid.py               # score formula, match_type, dedup, limit
│   ├── test_embedding_pipeline.py   # pipeline embed step, reembed, error swallow
│   └── test_cli_embeddings.py       # --embed-model, --semantic, --hybrid flags
├── benchmark.py                     # FTS vs Semantic vs Hybrid latency report
└── pyproject.toml
```

---

## Setup

```bash
# Requires uv (https://github.com/astral-sh/uv)
uv sync

# Verify installation
uv run ctx --help
```

### Optional: enable embeddings

Semantic and hybrid search require `sentence-transformers`:

```bash
uv sync --extra embeddings
```

If you manage dependencies with pip directly:

```bash
uv pip install sentence-transformers
# or plain pip:
pip install sentence-transformers
```

**Model weights** are downloaded from HuggingFace on first use and cached in
`~/.cache/huggingface/hub/`. No download happens on subsequent runs.

| Model | Size | Dimensions | Notes |
|-------|------|-----------|-------|
| `nomic-ai/nomic-embed-text-v1.5` | ~270 MB | 768 | Default. Best quality. Requires `einops` (bundled in `[embeddings]`) |
| `BAAI/bge-small-en-v1.5` | ~130 MB | 384 | Faster, lower memory, good for testing. No extra deps |

`uv sync --extra embeddings` installs both `sentence-transformers` and `einops` in one step.  
If you installed manually and hit `ImportError: einops not found`, run:

```bash
uv pip install einops
# or:
pip install einops
```

To verify the full install:

```bash
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print(model.encode(['hello']).shape)
"
# → (1, 384)
```

---

## Usage

### Initialise

```bash
uv run ctx init
# → creates context.db in the current directory

# Custom path:
uv run ctx init --db /path/to/myindex.db
CTX_DB=/path/to/myindex.db uv run ctx init
```

### Index a directory

```bash
uv run ctx index .
uv run ctx index /path/to/repo

# Tuning options:
uv run ctx index . --chunk-chars 2000 --overlap 5
uv run ctx index . --ignore "*.test.ts" --ignore "fixtures/**"
uv run ctx index . --verbose          # DEBUG logging
```

Subsequent runs are **incremental** — only new or modified files are re-indexed.

Example output:

```
╭─── Index complete ────╮
│ Files indexed:   217  │
│ Files deleted:     0  │
│ Files skipped:     0  │
│ Chunks created:  3048 │
│ Errors:            0  │
│ Duration:        12.4s│
╰───────────────────────╯
```

#### Generating embeddings during indexing

Pass `--embed-model` to also generate dense vector embeddings after chunking.  
Only new or changed chunks are embedded — unchanged chunks reuse their existing vectors.

```bash
# Index and embed in one pass:
uv run ctx index . --embed-model nomic-ai/nomic-embed-text-v1.5

# Supported models:
#   nomic-ai/nomic-embed-text-v1.5   (default, 768-dim, best quality)
#   BAAI/bge-small-en-v1.5           (384-dim, faster)

# Force full re-embedding:
uv run ctx index . --embed-model nomic-ai/nomic-embed-text-v1.5 --reembed

# Skip embedding even if model is configured:
uv run ctx index . --embed-model nomic-ai/nomic-embed-text-v1.5 --disable-embeddings
```

Output with embeddings:

```
╭─── Index complete ────────╮
│ Files indexed:   217      │
│ Chunks created:  3048     │
│ Chunks embedded: 3048     │
│ Duration:        47.2s    │
╰───────────────────────────╯
```

### Search

#### Lexical search (default — no model required)

```bash
uv run ctx search "jwt"
uv run ctx search "verify token" --limit 5
uv run ctx search "auth*" --path "%.py"    # restrict to Python files
uv run ctx search "jwt" --json             # machine-readable output
uv run ctx search "jwt" --all-chunks       # every matching chunk, files may repeat
```

BM25 query syntax:

| Syntax | Effect |
|--------|--------|
| `jwt` | Single-term match |
| `"jwt verify"` | Exact phrase |
| `jwt OR token` | Either term |
| `jwt AND NOT refresh` | Exclusion |
| `auth*` | Prefix wildcard |

Each file appears **at most once** by default (best-matching chunk). Use `--all-chunks` for all chunk-level hits.

#### Semantic search

Requires embeddings to have been generated with `ctx index --embed-model`.

```bash
uv run ctx search "token validation logic" --semantic
uv run ctx search "database retry" --semantic --embed-model BAAI/bge-small-en-v1.5
uv run ctx search "auth flow" --semantic --json
```

JSON output format:

```json
[
  {
    "path": "src/auth.py",
    "score": 0.912,
    "start_line": 14,
    "end_line": 38,
    "preview": "def verify_token(token: str) -> bool: ...",
    "match_type": "semantic"
  }
]
```

#### Hybrid search (recommended for best quality)

Combines BM25 and semantic signals:  
`final_score = 0.60 × lexical_norm + 0.40 × semantic_norm`

```bash
uv run ctx search "authentication" --hybrid
uv run ctx search "error handling" --hybrid --limit 20
uv run ctx search "config loader"  --hybrid --json
```

`match_type` in results is `"lexical"`, `"semantic"`, or `"hybrid"` — showing which signals found each chunk.

#### Force lexical fallback

`--disable-embeddings` overrides `--semantic`/`--hybrid` and forces FTS-only search:

```bash
uv run ctx search "jwt" --hybrid --disable-embeddings
# Equivalent to: ctx search "jwt"
```

### Stats

```bash
uv run ctx stats
```

```
╭──── Index stats ───────────────╮
│ Files indexed:     217         │
│ Chunks:           3048         │
│ Database size:    4.2 MB       │
│ Database path:  /repo/ctx.db   │
╰────────────────────────────────╯
```

### Reset

```bash
uv run ctx reset          # prompts for confirmation
uv run ctx reset --yes    # non-interactive
```

---

## Benchmark

Compare retrieval strategies against your own codebase:

```bash
uv run python benchmark.py [DIRECTORY]
```

Reports avg/P95 latency, results-per-query, and example output for each strategy:

```
Strategy        Avg latency    P95 latency    Results/query
──────────────────────────────────────────────────────────
FTS only        1.3 ms         2.1 ms         4.2
Semantic only   8.4 ms         11.2 ms        5.0
Hybrid          9.7 ms         13.0 ms        6.1
```

Semantic and hybrid rows are omitted if `sentence-transformers` is not installed.

---

## Supported file formats

### Plain text (always indexed)

Any file whose content is valid UTF-8 or latin-1 — `.py`, `.ts`, `.js`, `.go`, `.rs`, `.md`, `.yaml`, `.json`, `.toml`, `.sql`, `.sh`, and hundreds more.

### Rich documents (converted to Markdown via markitdown)

| Format | Extensions | What is extracted |
|--------|------------|-------------------|
| PDF | `.pdf` | Text layer; OCR fallback |
| Word | `.docx`, `.doc` | Paragraphs, tables |
| Excel | `.xlsx`, `.xls` | Sheet names, cell values |
| PowerPoint | `.pptx`, `.ppt` | Slide text |
| HTML | `.html`, `.htm` | Stripped markup |
| CSV | `.csv` | Rows as Markdown table |
| XML | `.xml` | Element text content |

### Skipped automatically

| Reason | Examples |
|--------|---------|
| Binary content (null-byte sniff) | compiled objects, images, audio |
| File size > 16 MB | very large data files |
| Compiled artefacts | `.pyc`, `.so`, `.dll`, `.exe` |
| Media files | `.png`, `.jpg`, `.mp4`, `.mp3` |
| Dependency directories | `node_modules/`, `.git/`, `__pycache__/`, `.venv/` |

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `CTX_DB` | `./context.db` | Path to the SQLite database |
| `CTX_LOG_LEVEL` | `30` (WARNING) | Numeric Python log level; `10` = DEBUG |
| `CTX_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Default embedding model for `--semantic`/`--hybrid` |

---

## Testing

```bash
uv run pytest                   # all tests
uv run pytest --cov=context_db  # with coverage report
uv run pytest -x -q             # fail-fast, quiet
```

Coverage target: **≥ 90%** (currently **98%**, 200+ tests).

Test modules:

| File | What it covers |
|------|----------------|
| `test_scanner.py` | Discovery, ignores, binary sniff |
| `test_fingerprint.py` | SHA-256, diff new/modified/deleted |
| `test_chunker.py` | Overlap, line ranges, validation |
| `test_extractor.py` | Rich formats (PDF, DOCX, XLSX, HTML, CSV) |
| `test_storage.py` | Migrations, upsert, cascade, FTS sync |
| `test_pipeline.py` | Full index, incremental, delete, progress callback |
| `test_search.py` | BM25 ranking, dedup, path filter, FTS operators |
| `test_cli.py` | All commands via CliRunner, env var, JSON output |
| `test_edge_cases.py` | Oversized files, latin-1 fallback, validators |
| `test_embedding_storage.py` | Migration 2, vector CRUD, cascade delete, serialisation |
| `test_embedder.py` | Embedder ABC, LocalEmbedder, lazy load, batch, retry |
| `test_semantic.py` | Cosine top-k (numpy + pure-python), end-to-end search |
| `test_hybrid.py` | Score formula, match_type, dedup, limit, FTS chunk search |
| `test_embedding_pipeline.py` | Pipeline embed step, reembed, error swallowing |
| `test_cli_embeddings.py` | `--embed-model`, `--semantic`, `--hybrid`, `--disable-embeddings` |

---

## Design principles

- **Local-first** — no network calls, no external services required.
- **Incremental by default** — SHA-256 hashing skips unchanged files; embedding step skips already-embedded chunks.
- **Embeddings are optional** — `ctx search` (FTS) works without any model; `--semantic`/`--hybrid` are opt-in.
- **No vector database** — embeddings are stored as float32 blobs in SQLite; cosine search is brute-force O(N·D), accelerated by numpy when available.
- **Atomic writes** — each file update is a single SQLite transaction; FTS and embedding cascade deletes keep state consistent.
- **File-level results by default** — lexical search deduplicates by file; semantic and hybrid are chunk-level.
- **Clean interfaces** — each subsystem has a single responsibility and well-defined Pydantic input/output types.

---

## Extension points

### Tree-sitter chunker

Replace `Chunker` with a `TreeSitterChunker` that emits function/class-level chunks with semantic metadata.

Hook point: `context_db/indexer/chunker.py` — `Chunker` is constructor-injected into `IndexingPipeline`.

### Custom embedding provider

Implement `Embedder` and register it in `create_embedder()`:

```python
# context_db/embeddings/providers/openai.py
from context_db.embeddings.embedder import Embedder

class OpenAIEmbedder(Embedder):
    @property
    def model_name(self) -> str: return "text-embedding-3-small"

    @property
    def dimensions(self) -> int: return 1536

    def embed(self, text: str) -> list[float]:
        import openai
        return openai.embeddings.create(input=text, model=self.model_name).data[0].embedding
```

### MCP server

Expose all three retrieval strategies as MCP tools:

```python
# mcp_server.py
from mcp import Server
from context_db.retrieval.search import SearchEngine
from context_db.retrieval.hybrid import hybrid_search
from context_db.storage.db import open_db

server = Server("context-db")

@server.tool("search_codebase")
async def search_codebase(query: str, mode: str = "hybrid") -> list[dict]:
    conn = open_db(Path("context.db"))
    if mode == "hybrid":
        results = hybrid_search(query, embedder=embedder, repo=repo, conn=conn)
    else:
        results = SearchEngine(conn).search(query)
    return [r.model_dump() for r in results]
```

The repository pattern and Pydantic models make all result types trivially serialisable over MCP transport.

### Cross-session context reuse

Add a `sessions` table and `session_queries` log — track which chunks were used in each agent conversation, then weight frequently-retrieved chunks higher in future rankings.

Hook point: `Repository` + a new `FeedbackRepository`.
