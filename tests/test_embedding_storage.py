"""Tests for embedding storage: migration, repository methods, cascade."""

from __future__ import annotations

from pathlib import Path

import pytest

from context_db.models import Chunk, FileMetadata
from context_db.storage.repository import Repository, _blob_to_vec, _vec_to_blob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(path: Path, hash_: str = "a" * 64, mtime: float = 1000.0) -> FileMetadata:
    return FileMetadata(path=path, hash=hash_, mtime=mtime, size=10)


def _make_chunk_with_embedding(
    repo: Repository,
    path: Path = Path("/src/f.py"),
    content: str = "hello world",
    model: str = "test-model",
    vector: list[float] | None = None,
) -> tuple[int, list[float]]:
    """Insert a file + chunk, optionally store an embedding; return (chunk_id, vector)."""
    file_id = repo.upsert_file(_meta(path))
    repo.insert_chunks(
        file_id,
        [Chunk(path=path, start_line=1, end_line=1, content=content)],
    )
    stored = repo.get_chunks_for_file(file_id)
    chunk_id = stored[0].id
    vec = vector if vector is not None else [0.1, 0.2, 0.3]
    repo.upsert_embedding(chunk_id, vec, model)
    return chunk_id, vec


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


class TestVectorSerialisation:
    def test_roundtrip_float32(self) -> None:
        original = [0.1, 0.5, -0.9, 1.0]
        blob = _vec_to_blob(original)
        recovered = _blob_to_vec(blob)
        assert len(recovered) == len(original)
        for a, b in zip(original, recovered):
            assert abs(a - b) < 1e-6

    def test_empty_vector(self) -> None:
        assert _blob_to_vec(_vec_to_blob([])) == []

    def test_blob_is_bytes(self) -> None:
        blob = _vec_to_blob([1.0, 2.0])
        assert isinstance(blob, bytes)
        # float32 → 4 bytes each
        assert len(blob) == 8


# ---------------------------------------------------------------------------
# Migration: chunk_embeddings table exists
# ---------------------------------------------------------------------------


class TestMigration:
    def test_chunk_embeddings_table_exists(self, tmp_repo: tuple) -> None:
        _, conn, _ = tmp_repo
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "chunk_embeddings" in tables

    def test_schema_version_is_2(self, tmp_repo: tuple) -> None:
        _, conn, _ = tmp_repo
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version == 2

    def test_model_index_exists(self, tmp_repo: tuple) -> None:
        _, conn, _ = tmp_repo
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_chunk_embeddings_model" in indexes


# ---------------------------------------------------------------------------
# upsert_embedding
# ---------------------------------------------------------------------------


class TestUpsertEmbedding:
    def test_insert_embedding(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        chunk_id, vec = _make_chunk_with_embedding(repo)
        row = conn.execute(
            "SELECT chunk_id, dimensions, model FROM chunk_embeddings WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        assert row is not None
        assert row["chunk_id"] == chunk_id
        assert row["dimensions"] == len(vec)
        assert row["model"] == "test-model"

    def test_upsert_overwrites_on_conflict(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        chunk_id, _ = _make_chunk_with_embedding(repo, vector=[0.1, 0.2, 0.3])
        new_vec = [0.9, 0.8, 0.7]
        repo.upsert_embedding(chunk_id, new_vec, "test-model")
        pairs = repo.get_all_embeddings("test-model")
        assert len(pairs) == 1
        cid, recovered = pairs[0]
        assert cid == chunk_id
        for a, b in zip(new_vec, recovered):
            assert abs(a - b) < 1e-6

    def test_created_at_is_set(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        chunk_id, _ = _make_chunk_with_embedding(repo)
        row = conn.execute(
            "SELECT created_at FROM chunk_embeddings WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        assert row["created_at"] > 0


# ---------------------------------------------------------------------------
# upsert_embeddings_batch
# ---------------------------------------------------------------------------


class TestUpsertEmbeddingsBatch:
    def test_batch_inserts_multiple(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        p = Path("/src/batch.py")
        file_id = repo.upsert_file(_meta(p))
        chunks = [
            Chunk(path=p, start_line=i, end_line=i, content=f"line {i}")
            for i in range(1, 4)
        ]
        repo.insert_chunks(file_id, chunks)
        stored = repo.get_chunks_for_file(file_id)
        chunk_ids = [c.id for c in stored]
        vectors = [[float(i), float(i + 1)] for i in range(3)]

        repo.upsert_embeddings_batch(chunk_ids, vectors, "batch-model")

        pairs = dict(repo.get_all_embeddings("batch-model"))
        assert set(pairs.keys()) == set(chunk_ids)

    def test_batch_empty(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        repo.upsert_embeddings_batch([], [], "batch-model")
        assert repo.get_embedding_count("batch-model") == 0


# ---------------------------------------------------------------------------
# get_all_embeddings
# ---------------------------------------------------------------------------


class TestGetAllEmbeddings:
    def test_returns_correct_pairs(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        vec = [0.1, 0.2, 0.3]
        chunk_id, _ = _make_chunk_with_embedding(repo, vector=vec)
        pairs = repo.get_all_embeddings("test-model")
        assert len(pairs) == 1
        cid, recovered = pairs[0]
        assert cid == chunk_id
        for a, b in zip(vec, recovered):
            assert abs(a - b) < 1e-6

    def test_filters_by_model(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        p1, p2 = Path("/a.py"), Path("/b.py")
        file_id1 = repo.upsert_file(_meta(p1))
        repo.insert_chunks(file_id1, [Chunk(path=p1, start_line=1, end_line=1, content="a")])
        c1 = repo.get_chunks_for_file(file_id1)[0]

        file_id2 = repo.upsert_file(_meta(p2))
        repo.insert_chunks(file_id2, [Chunk(path=p2, start_line=1, end_line=1, content="b")])
        c2 = repo.get_chunks_for_file(file_id2)[0]

        repo.upsert_embedding(c1.id, [1.0], "model-a")
        repo.upsert_embedding(c2.id, [2.0], "model-b")

        assert len(repo.get_all_embeddings("model-a")) == 1
        assert len(repo.get_all_embeddings("model-b")) == 1
        assert repo.get_all_embeddings("model-c") == []


# ---------------------------------------------------------------------------
# get_chunks_without_embeddings
# ---------------------------------------------------------------------------


class TestGetChunksWithoutEmbeddings:
    def test_returns_un_embedded_chunks(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        p = Path("/src/partial.py")
        file_id = repo.upsert_file(_meta(p))
        chunks = [
            Chunk(path=p, start_line=1, end_line=1, content="chunk A"),
            Chunk(path=p, start_line=2, end_line=2, content="chunk B"),
        ]
        repo.insert_chunks(file_id, chunks)
        stored = repo.get_chunks_for_file(file_id)

        # Only embed the first chunk.
        repo.upsert_embedding(stored[0].id, [0.1, 0.2], "m")

        missing = repo.get_chunks_without_embeddings("m")
        assert len(missing) == 1
        assert missing[0].id == stored[1].id

    def test_returns_all_when_none_embedded(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        p = Path("/src/none.py")
        file_id = repo.upsert_file(_meta(p))
        repo.insert_chunks(file_id, [Chunk(path=p, start_line=1, end_line=1, content="x")])
        missing = repo.get_chunks_without_embeddings("any-model")
        assert len(missing) == 1

    def test_returns_empty_when_all_embedded(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        chunk_id, _ = _make_chunk_with_embedding(repo)
        missing = repo.get_chunks_without_embeddings("test-model")
        assert missing == []


# ---------------------------------------------------------------------------
# delete_embeddings_for_model
# ---------------------------------------------------------------------------


class TestDeleteEmbeddingsForModel:
    def test_deletes_only_matching_model(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        p1, p2 = Path("/x.py"), Path("/y.py")
        fid1 = repo.upsert_file(_meta(p1))
        fid2 = repo.upsert_file(_meta(p2))
        repo.insert_chunks(fid1, [Chunk(path=p1, start_line=1, end_line=1, content="x")])
        repo.insert_chunks(fid2, [Chunk(path=p2, start_line=1, end_line=1, content="y")])
        c1 = repo.get_chunks_for_file(fid1)[0]
        c2 = repo.get_chunks_for_file(fid2)[0]

        repo.upsert_embedding(c1.id, [1.0], "model-del")
        repo.upsert_embedding(c2.id, [2.0], "model-keep")

        repo.delete_embeddings_for_model("model-del")

        assert repo.get_all_embeddings("model-del") == []
        assert len(repo.get_all_embeddings("model-keep")) == 1


# ---------------------------------------------------------------------------
# get_embedding_count
# ---------------------------------------------------------------------------


class TestGetEmbeddingCount:
    def test_count_increments(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        assert repo.get_embedding_count("m") == 0
        _make_chunk_with_embedding(repo, path=Path("/a.py"), model="m")
        assert repo.get_embedding_count("m") == 1

    def test_count_zero_for_unknown_model(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        assert repo.get_embedding_count("no-such-model") == 0


# ---------------------------------------------------------------------------
# Cascade delete: deleting a chunk removes its embedding
# ---------------------------------------------------------------------------


class TestCascadeDelete:
    def test_embedding_deleted_when_chunk_deleted(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        chunk_id, _ = _make_chunk_with_embedding(repo)
        assert repo.get_embedding_count("test-model") == 1

        conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        conn.commit()

        assert repo.get_embedding_count("test-model") == 0

    def test_embedding_deleted_when_file_deleted(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        p = Path("/cascade.py")
        chunk_id, _ = _make_chunk_with_embedding(repo, path=p)
        assert repo.get_embedding_count("test-model") == 1

        repo.delete_file_transactional(p)

        assert repo.get_embedding_count("test-model") == 0
