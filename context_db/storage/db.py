"""SQLite connection management and schema migrations.

The database lives in a single file whose path is determined at runtime (via
the CLI or programmatically).  All schema changes are expressed as numbered
migrations that run in order; already-applied migrations are skipped.

Schema
------
files
    id        INTEGER PRIMARY KEY
    path      TEXT UNIQUE NOT NULL
    hash      TEXT NOT NULL        -- hex SHA-256
    mtime     REAL NOT NULL        -- Unix timestamp

chunks
    id        INTEGER PRIMARY KEY
    file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE
    start_line INTEGER NOT NULL
    end_line   INTEGER NOT NULL
    content    TEXT NOT NULL

chunks_fts   (FTS5 virtual table)
    content   TEXT                  -- mirrors chunks.content
    tokenize  'porter unicode61'    -- stemming + unicode normalisation

chunk_embeddings
    chunk_id   INTEGER PRIMARY KEY  -- FK → chunks(id) ON DELETE CASCADE
    embedding  BLOB NOT NULL        -- float32 array, little-endian
    model      TEXT NOT NULL        -- e.g. "nomic-ai/nomic-embed-text-v1.5"
    dimensions INTEGER NOT NULL
    created_at INTEGER NOT NULL     -- Unix timestamp (int seconds)

schema_version
    version   INTEGER PRIMARY KEY   -- monotonic migration counter
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------
# Each migration is a list of SQL statements executed inside a single
# transaction.  Add new migrations by appending to this list — never edit
# existing entries.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[list[str]] = [
    # ── Migration 0: initial schema ──────────────────────────────────────
    [
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS files (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            path  TEXT    UNIQUE NOT NULL,
            hash  TEXT    NOT NULL,
            mtime REAL    NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            start_line INTEGER NOT NULL,
            end_line   INTEGER NOT NULL,
            content    TEXT    NOT NULL
        )
        """,
        # FTS5 content table shadowing chunks.content.
        # Using content='chunks' lets FTS read row content on demand while
        # keeping the index compact.
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            content='chunks',
            content_rowid='id',
            tokenize='porter unicode61'
        )
        """,
        # Triggers to keep the FTS index in sync with the chunks table.
        """
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content)
            VALUES (new.id, new.content);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
            INSERT INTO chunks_fts(rowid, content)
            VALUES (new.id, new.content);
        END
        """,
        # Speed up look-ups by file_id (used heavily during re-indexing).
        "CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)",
        # Record that migration 0 ran.
        "INSERT OR IGNORE INTO schema_version(version) VALUES (0)",
    ],
    # ── Migration 1: FTS5 vocabulary table for autocomplete ──────────────
    [
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_vocab USING fts5vocab(
            'chunks_fts', 'row'
        )
        """,
        "INSERT OR IGNORE INTO schema_version(version) VALUES (1)",
    ],
    # ── Migration 2: per-chunk vector embeddings ──────────────────────────
    [
        """
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id   INTEGER PRIMARY KEY,
            embedding  BLOB    NOT NULL,
            model      TEXT    NOT NULL,
            dimensions INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        )
        """,
        # Speeds up "all embeddings for model X" scans used by semantic search.
        "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings(model)",
        "INSERT OR IGNORE INTO schema_version(version) VALUES (2)",
    ],
]


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply performance and safety pragmas to a freshly opened connection."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 134217728")  # 128 MiB


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database and run pending migrations.

    Returns a connection with ``row_factory = sqlite3.Row`` set so that rows
    can be accessed by column name.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    _migrate(conn)
    logger.debug("db_open", path=str(db_path))
    return conn


def _current_version(conn: sqlite3.Connection) -> int:
    """Return the highest migration version recorded, or -1 if none."""
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0]) if row[0] is not None else -1
    except sqlite3.OperationalError:
        # schema_version table doesn't exist yet.
        return -1


def _migrate(conn: sqlite3.Connection) -> None:
    """Run any migrations that haven't been applied yet."""
    current = _current_version(conn)
    pending = [m for i, m in enumerate(_MIGRATIONS) if i > current]
    if not pending:
        return

    for i, statements in enumerate(pending, start=current + 1):
        logger.info("db_migration", version=i)
        with conn:
            for sql in statements:
                conn.execute(sql)

    logger.info("db_migrations_done", applied=len(pending))
