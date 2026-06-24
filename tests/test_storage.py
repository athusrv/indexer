"""Tests for context_db.storage (db + repository)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from context_db.models import Chunk, FileMetadata
from context_db.storage.db import open_db
from context_db.storage.repository import Repository


def _meta(path: Path, hash_: str = "a" * 64, mtime: float = 1000.0) -> FileMetadata:
    return FileMetadata(path=path, hash=hash_, mtime=mtime, size=10)


class TestOpenDb:
    def test_creates_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = open_db(db_path)
        conn.close()
        assert db_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "deep" / "test.db"
        conn = open_db(db_path)
        conn.close()
        assert db_path.exists()

    def test_idempotent_migrations(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn1 = open_db(db_path)
        conn1.close()
        # Open a second time — should not raise.
        conn2 = open_db(db_path)
        conn2.close()

    def test_foreign_keys_enabled(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = open_db(db_path)
        result = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        conn.close()
        assert result == 1

    def test_tables_created(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = open_db(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "files" in tables
        assert "chunks" in tables
        assert "schema_version" in tables


class TestRepository:
    def test_upsert_file_returns_id(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        meta = _meta(Path("/src/auth.py"))
        file_id = repo.upsert_file(meta)
        assert isinstance(file_id, int)
        assert file_id > 0

    def test_upsert_file_idempotent(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        meta = _meta(Path("/src/auth.py"))
        id1 = repo.upsert_file(meta)
        id2 = repo.upsert_file(meta)
        assert id1 == id2

    def test_upsert_updates_hash(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        p = Path("/src/changed.py")
        repo.upsert_file(_meta(p, hash_="a" * 64))
        repo.upsert_file(_meta(p, hash_="b" * 64))
        stored = repo.get_file_by_path(p)
        assert stored is not None
        assert stored.hash == "b" * 64

    def test_get_file_by_path_none_for_missing(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        assert repo.get_file_by_path(Path("/nonexistent.py")) is None

    def test_insert_and_get_chunks(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        meta = _meta(Path("/src/f.py"))
        file_id = repo.upsert_file(meta)
        chunks = [
            Chunk(path=Path("/src/f.py"), start_line=1, end_line=5, content="chunk 1"),
            Chunk(path=Path("/src/f.py"), start_line=6, end_line=10, content="chunk 2"),
        ]
        repo.insert_chunks(file_id, chunks)
        stored = repo.get_chunks_for_file(file_id)
        assert len(stored) == 2
        assert stored[0].start_line == 1

    def test_delete_file_cascades_chunks(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        p = Path("/src/todelete.py")
        file_id = repo.upsert_file(_meta(p))
        repo.insert_chunks(file_id, [
            Chunk(path=p, start_line=1, end_line=2, content="x")
        ])
        repo.delete_file(p)
        assert repo.get_file_by_path(p) is None
        assert repo.get_chunks_for_file(file_id) == []

    def test_replace_file_chunks_atomically(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        p = Path("/src/replace.py")
        meta = _meta(p)
        chunks_v1 = [Chunk(path=p, start_line=1, end_line=3, content="version 1")]
        repo.replace_file_chunks(meta, chunks_v1)

        chunks_v2 = [Chunk(path=p, start_line=1, end_line=5, content="version 2")]
        repo.replace_file_chunks(meta, chunks_v2)

        file = repo.get_file_by_path(p)
        stored_chunks = repo.get_chunks_for_file(file.id)
        assert len(stored_chunks) == 1
        assert stored_chunks[0].content == "version 2"

    def test_get_all_file_metadata(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        repo.upsert_file(_meta(Path("/a.py")))
        repo.upsert_file(_meta(Path("/b.py")))
        all_meta = repo.get_all_file_metadata()
        assert len(all_meta) == 2

    def test_stats(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        p = Path("/src/stats.py")
        file_id = repo.upsert_file(_meta(p))
        repo.insert_chunks(file_id, [
            Chunk(path=p, start_line=1, end_line=2, content="stat chunk")
        ])
        stats = repo.get_stats(db_path)
        assert stats.file_count == 1
        assert stats.chunk_count == 1
        assert stats.db_size_bytes > 0

    def test_reset_clears_data(self, tmp_repo: tuple) -> None:
        repo, conn, db_path = tmp_repo
        p = Path("/src/reset.py")
        file_id = repo.upsert_file(_meta(p))
        repo.insert_chunks(file_id, [Chunk(path=p, start_line=1, end_line=1, content="x")])
        repo.reset()
        assert repo.get_all_file_metadata() == {}
