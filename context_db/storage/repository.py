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

import array as _array
import sqlite3
import time
from pathlib import Path

import structlog

from context_db.models import (
    Chunk,
    ChunkFileInfo,
    FileMetadata,
    IndexStats,
    StoredChunk,
    StoredFile,
)


# ---------------------------------------------------------------------------
# Vector serialisation helpers (stdlib only, float32 little-endian)
# ---------------------------------------------------------------------------


def _vec_to_blob(vector: list[float]) -> bytes:
    return _array.array("f", vector).tobytes()


def _blob_to_vec(blob: bytes) -> list[float]:
    arr = _array.array("f")
    arr.frombytes(blob)
    return list(arr)

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

    def get_chunk_file_info(self, chunk_ids: list[int]) -> list[ChunkFileInfo]:
        """Return chunk data joined with file path for the given chunk ids."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"""
            SELECT c.id, c.start_line, c.end_line, c.content, f.path
            FROM   chunks c
            JOIN   files  f ON c.file_id = f.id
            WHERE  c.id IN ({placeholders})
            """,  # noqa: S608
            chunk_ids,
        ).fetchall()
        return [
            ChunkFileInfo(
                chunk_id=row["id"],
                path=Path(row["path"]),
                start_line=row["start_line"],
                end_line=row["end_line"],
                content=row["content"],
            )
            for row in rows
        ]

    def get_chunks_by_ids(self, chunk_ids: list[int]) -> list[StoredChunk]:
        """Fetch specific chunks by their primary keys (order not guaranteed)."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT id, file_id, start_line, end_line, content FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
            chunk_ids,
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

    # ------------------------------------------------------------------
    # Embedding operations
    # ------------------------------------------------------------------

    def upsert_embedding(
        self,
        chunk_id: int,
        vector: list[float],
        model: str,
    ) -> None:
        """Insert or replace the embedding for a single chunk."""
        blob = _vec_to_blob(vector)
        self._conn.execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, embedding, model, dimensions, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                embedding  = excluded.embedding,
                model      = excluded.model,
                dimensions = excluded.dimensions,
                created_at = excluded.created_at
            """,
            (chunk_id, blob, model, len(vector), int(time.time())),
        )

    def upsert_embeddings_batch(
        self,
        chunk_ids: list[int],
        vectors: list[list[float]],
        model: str,
    ) -> None:
        """Insert or replace embeddings for multiple chunks in one transaction."""
        now = int(time.time())
        self._conn.executemany(
            """
            INSERT INTO chunk_embeddings (chunk_id, embedding, model, dimensions, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                embedding  = excluded.embedding,
                model      = excluded.model,
                dimensions = excluded.dimensions,
                created_at = excluded.created_at
            """,
            [
                (cid, _vec_to_blob(vec), model, len(vec), now)
                for cid, vec in zip(chunk_ids, vectors)
            ],
        )

    def get_all_embeddings(self, model: str) -> list[tuple[int, list[float]]]:
        """Return all ``(chunk_id, vector)`` pairs stored for *model*."""
        rows = self._conn.execute(
            "SELECT chunk_id, embedding FROM chunk_embeddings WHERE model = ?",
            (model,),
        ).fetchall()
        return [(row["chunk_id"], _blob_to_vec(row["embedding"])) for row in rows]

    def get_chunks_without_embeddings(self, model: str) -> list[StoredChunk]:
        """Return chunks that have no embedding for *model* yet."""
        rows = self._conn.execute(
            """
            SELECT c.id, c.file_id, c.start_line, c.end_line, c.content
            FROM   chunks c
            LEFT   JOIN chunk_embeddings e
                   ON  e.chunk_id = c.id
                   AND e.model    = ?
            WHERE  e.chunk_id IS NULL
            ORDER  BY c.id
            """,
            (model,),
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

    def delete_embeddings_for_model(self, model: str) -> None:
        """Delete all stored embeddings for *model* (used by ``--reembed``)."""
        self._conn.execute(
            "DELETE FROM chunk_embeddings WHERE model = ?",
            (model,),
        )
        logger.info("embeddings_deleted", model=model)

    def get_embedding_count(self, model: str) -> int:
        """Return the number of stored embeddings for *model*."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE model = ?",
            (model,),
        ).fetchone()
        return int(row[0])
