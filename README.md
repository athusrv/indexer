# context-db

A **production-quality, embedding-free filesystem indexing engine** for AI agents.

Converts a local codebase into a persistent, BM25-searchable SQLite database so language models can retrieve relevant code context without re-exploring the repository on every request.

```
ctx index .          # index a codebase
ctx search "jwt"     # retrieve relevant chunks instantly
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLI  (Typer)                             │
│   ctx init │ ctx index │ ctx search │ ctx stats │ ctx reset      │
└───────────────────────┬──────────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │    Indexing Pipeline      │
          │  scan → diff → chunk      │
          │  → persist (atomic)       │
          └──────┬──────────┬─────────┘
                 │          │
    ┌────────────▼──┐  ┌────▼───────────────┐
    │    Scanner    │  │   Fingerprinter     │
    │  os.walk +    │  │  SHA-256 hashing    │
    │  ignore rules │  │  change detection   │
    └───────────────┘  └────────────────────┘
                 │
    ┌────────────▼──────────┐
    │       Chunker         │
    │  line-aligned slices  │
    │  configurable overlap │
    └────────────┬──────────┘
                 │
    ┌────────────▼──────────────────────────┐
    │         Storage Layer                 │
    │                                       │
    │  SQLite  ┌──────────┐  ┌──────────┐   │
    │          │  files   │  │  chunks  │   │
    │          └──────────┘  └────┬─────┘   │
    │                             │ trigger │
    │                        ┌────▼──────┐  │
    │                        │chunks_fts │  │
    │                        │  (FTS5)   │  │
    │                        └───────────┘  │
    └───────────────────────────┬───────────┘
                                │
    ┌───────────────────────────▼───────────┐
    │          Search Engine                │
    │   FTS5 MATCH + bm25() ranking        │
    └───────────────────────────────────────┘
```

### Data Flow

```
Filesystem
    │
    ▼ Scanner          → [DiscoveredFile, ...]
    │
    ▼ Fingerprinter    → ChangeSet (new / modified / deleted)
    │
    ▼ File Read        → raw text (UTF-8 → latin-1 fallback)
    │
    ▼ Chunker          → [Chunk(path, start_line, end_line, content), ...]
    │
    ▼ Repository       → SQLite  (transactional upsert + FTS5 trigger sync)
```

### SQLite Schema

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
```

Triggers keep `chunks_fts` in sync with `chunks` automatically (INSERT / UPDATE / DELETE).

---

## Project Layout

```
context-db/
├── context_db/
│   ├── models.py            # Pydantic data models (shared)
│   ├── cli.py               # Typer CLI entry-point
│   ├── indexer/
│   │   ├── scanner.py       # Filesystem traversal
│   │   ├── fingerprint.py   # SHA-256 hashing + change detection
│   │   ├── chunker.py       # Line-aligned character chunking
│   │   └── pipeline.py      # Orchestration: scan→diff→chunk→persist
│   ├── storage/
│   │   ├── db.py            # SQLite connection + migrations
│   │   └── repository.py    # Repository pattern (all SQL here)
│   └── retrieval/
│       └── search.py        # FTS5 + BM25 search engine
├── tests/
│   ├── conftest.py
│   ├── test_scanner.py
│   ├── test_fingerprint.py
│   ├── test_chunker.py
│   ├── test_storage.py
│   ├── test_pipeline.py
│   ├── test_search.py
│   ├── test_cli.py
│   └── test_edge_cases.py
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
# Requires uv (https://github.com/astral-sh/uv)
uv sync

# Verify installation
uv run ctx --help
```

---

## Usage

### Initialise

```bash
uv run ctx init
# → creates context.db in the current directory
```

Custom database path:

```bash
uv run ctx init --db /path/to/myindex.db
# or via env var:
CTX_DB=/path/to/myindex.db uv run ctx init
```

### Index a directory

```bash
uv run ctx index .
uv run ctx index /path/to/repo

# With options:
uv run ctx index . --chunk-chars 2000 --overlap 5
uv run ctx index . --ignore "*.test.ts" --ignore "fixtures/**"
uv run ctx index . --verbose          # enable DEBUG logging
```

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

Subsequent runs are **incremental** — only new or modified files are re-indexed.

### Search

```bash
uv run ctx search "jwt"
uv run ctx search "verify token" --limit 5
uv run ctx search "auth*" --path "%.py"   # restrict to Python files
uv run ctx search "jwt" --json            # machine-readable output
```

The search engine uses SQLite's native **BM25** ranking, which prefers chunks with higher term density and rarity. Queries support FTS5 operators:

| Syntax | Effect |
|--------|--------|
| `jwt`  | Single-term match |
| `"jwt verify"` | Exact phrase |
| `jwt OR token` | Either term |
| `jwt AND NOT refresh` | Exclusion |
| `auth*` | Prefix wildcard |

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

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `CTX_DB` | `./context.db` | Path to the SQLite database |
| `CTX_LOG_LEVEL` | `30` (WARNING) | Numeric Python log level; set to `10` for DEBUG |

---

## Testing

```bash
uv run pytest                              # all tests
uv run pytest --cov=context_db            # with coverage
uv run pytest -x -q                       # fail-fast
```

Coverage target: **≥ 90%** (currently 97%).

---

## Supported File Formats

### Plain text (always indexed)

Any file whose content is valid UTF-8 or latin-1 text — `.py`, `.ts`, `.js`,
`.go`, `.rs`, `.md`, `.yaml`, `.json`, `.toml`, `.sql`, `.sh`, and hundreds more.

### Rich documents (converted via markitdown)

| Format | Extensions | What is extracted |
|--------|------------|-------------------|
| PDF | `.pdf` | Text layer; OCR fallback |
| Word | `.docx`, `.doc` | Paragraphs, tables |
| Excel | `.xlsx`, `.xls` | Sheet names, cell values |
| PowerPoint | `.pptx`, `.ppt` | Slide text |
| HTML | `.html`, `.htm` | Stripped markup (Markdown) |
| CSV | `.csv` | Rows as Markdown table |
| XML | `.xml` | Element text content |

All rich formats are converted to Markdown before chunking, so headings,
bullet points, and table structure are preserved in the search index.

### Skipped

Binary files (null-byte sniff), files > 4 MB, compiled artefacts
(`.pyc`, `.so`, `.dll`, …), media (`.png`, `.mp4`, …), and dependency
directories (`node_modules/`, `.git/`, `__pycache__/`, …).

---

## Design Principles

- **Local-first** — no network calls, no external services.
- **Embedding-free** — pure BM25 keyword search via SQLite FTS5.
- **Incremental by default** — SHA-256 hashing skips unchanged files.
- **Atomic writes** — each file update is a single SQLite transaction.
- **Clean interfaces** — each subsystem has a single responsibility and well-defined input/output types.

---

## Extension Points

### Tree-sitter integration

Replace `Chunker` with a `TreeSitterChunker` that:
- Parses AST before splitting
- Emits function/class-level chunks with semantic metadata
- Attaches symbol names, docstrings, and import graphs to each chunk

Hook point: `context_db/indexer/chunker.py` — `Chunker` is injected into `IndexingPipeline`.

### Semantic / hybrid retrieval

Add an optional `VectorIndex` alongside `chunks_fts`:
- Embed chunks with a local model (e.g. `nomic-embed-text` via `llama.cpp`)
- Store vectors in a separate SQLite table or `sqlite-vss` extension
- Merge BM25 + cosine scores via Reciprocal Rank Fusion in `SearchEngine`

Hook point: `context_db/retrieval/search.py` — `SearchEngine.search()` can be extended to call a parallel vector path and merge results.

### Cross-session context reuse

Add a `sessions` table and `session_queries` log:
- Track which chunks were used in each agent conversation
- Weight frequently-retrieved chunks higher in future rankings
- Persist relevance feedback across sessions

Hook point: `Repository.get_stats()` and a new `FeedbackRepository`.

### MCP server integration

Expose `SearchEngine.search()` as an MCP tool:

```python
# mcp_server.py (future)
from mcp import Server
from context_db.retrieval.search import SearchEngine
from context_db.storage.db import open_db

server = Server("context-db")

@server.tool("search_codebase")
async def search_codebase(query: str, limit: int = 10) -> list[dict]:
    conn = open_db(Path("context.db"))
    engine = SearchEngine(conn)
    results = engine.search(query, limit=limit)
    return [r.model_dump() for r in results]
```

The repository pattern and Pydantic models make this trivially serialisable over MCP transport.
