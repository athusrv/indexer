"""Repository pattern over the SQLite storage layer.

All SQL lives here.  Higher-level subsystems (pipeline, search) interact only
with this class and never touch raw SQLite directly.

Transactional contract
----------------------
Methods that mutate state accept an *optional* ``conn`` parameter so callers
can compose multiple operations inside a single transaction.  When no external
connection is provided the repository uses its own managed connection.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

from context_db.models import (
    Chunk,
    FileMetadata,
    IndexStats,
    StoredChunk,
    StoredFile,
)

logger = structlog.get_logger(__name__)


class Repository:
    """High-level data access object for the context-db database.

    Parameters
    ----------
    conn:
        An open :class:`sqlite3.Connection` produced by
        :func:`context_db.storage.db.open_db`.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def get_all_file_metadata(self) -> dict[str, FileMetadata]:
        """Return all stored files as a ``{path_str: FileMetadata}`` mapping."""
        rows = self._conn.execute(
            "SELECT id, path, hash, mtime FROM files"
        ).fetchall()
        result: dict[str, FileMetadata] = {}
        for row in rows:
            meta = FileMetadata(
                path=Path(row["path"]),
                hash=row["hash"],
                mtime=row["mtime"],
                size=0,  # size not stored; set to 0 for stored records
            )
            result[row["path"]] = meta
        return result

    def upsert_file(self, meta: FileMetadata) -> int:
        """Insert or update a file record.  Returns the row's ``id``."""
        cur = self._conn.execute(
            """
            INSERT INTO files (path, hash, mtime)
            VALUES (:path, :hash, :mtime)
            ON CONFLICT(path) DO UPDATE SET
                hash  = excluded.hash,
                mtime = excluded.mtime
            RETURNING id
            """,
            {
                "path": str(meta.path),
                "hash": meta.hash,
                "mtime": meta.mtime,
            },
        )
        row = cur.fetchone()
        return int(row["id"])

    def delete_file(self, path: Path) -> None:
        """Delete a file and all its chunks (cascade handles chunks)."""
        self._conn.execute(
            "DELETE FROM files WHERE path = ?",
            (str(path),),
        )

    def get_file_by_path(self, path: Path) -> StoredFile | None:
        """Fetch a stored file by its absolute path, or ``None`` if absent."""
        row = self._conn.execute(
            "SELECT id, path, hash, mtime FROM files WHERE path = ?",
            (str(path),),
        ).fetchone()
        if row is None:
            return None
        return StoredFile(
            id=row["id"],
            path=Path(row["path"]),
            hash=row["hash"],
            mtime=row["mtime"],
        )

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def insert_chunks(self, file_id: int, chunks: list[Chunk]) -> None:
        """Bulk-insert chunks for a file (FTS triggers handle indexing)."""
        self._conn.executemany(
            """
            INSERT INTO chunks (file_id, start_line, end_line, content)
            VALUES (:file_id, :start_line, :end_line, :content)
            """,
            [
                {
                    "file_id": file_id,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "content": c.content,
                }
                for c in chunks
            ],
        )

    def delete_chunks_for_file(self, file_id: int) -> None:
        """Delete all chunks belonging to *file_id* (FTS triggers handle sync)."""
        self._conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    def get_chunks_for_file(self, file_id: int) -> list[StoredChunk]:
        """Return all chunks for a given file id, ordered by start_line."""
        rows = self._conn.execute(
            """
            SELECT id, file_id, start_line, end_line, content
            FROM chunks
            WHERE file_id = ?
            ORDER BY start_line
            """,
            (file_id,),
        ).fetchall()
        return [
            StoredChunk(
                id=row["id"],
                file_id=row["file_id"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                content=row["content"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Transactional helpers
    # ------------------------------------------------------------------

    def replace_file_chunks(
        self,
        meta: FileMetadata,
        chunks: list[Chunk],
    ) -> None:
        """Atomically upsert file metadata and replace all its chunks.

        This is the core write path used by the indexing pipeline.  The
        entire operation runs inside a single transaction so the DB never
        holds a partially-updated file.
        """
        with self._conn:
            file_id = self.upsert_file(meta)
            self.delete_chunks_for_file(file_id)
            if chunks:
                self.insert_chunks(file_id, chunks)
        logger.debug(
            "file_replaced",
            path=str(meta.path),
            file_id=file_id,
            chunks=len(chunks),
        )

    def delete_file_transactional(self, path: Path) -> None:
        """Delete a file and its chunks inside a transaction."""
        with self._conn:
            self.delete_file(path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, db_path: Path) -> IndexStats:
        """Return aggregate statistics about the current index."""
        file_count = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        db_size = db_path.stat().st_size if db_path.exists() else 0
        return IndexStats(
            file_count=file_count,
            chunk_count=chunk_count,
            db_size_bytes=db_size,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Truncate all data tables and rebuild the FTS index."""
        with self._conn:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM files")
            # Rebuild the FTS shadow tables.
            self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        logger.info("repository_reset")

    def optimize_fts(self) -> None:
        """Merge FTS index segments for faster query performance."""
        self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        logger.debug("fts_optimized")
