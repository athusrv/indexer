"""Tests for context_db.retrieval.semantic."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from context_db.embeddings.embedder import Embedder
from context_db.models import Chunk, FileMetadata
from context_db.retrieval.semantic import (
    SemanticHit,
    _cosine_top_k_numpy,
    _cosine_top_k_pure,
    semantic_search,
)
from context_db.storage.repository import Repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(path: Path, hash_: str = "a" * 64) -> FileMetadata:
    return FileMetadata(path=path, hash=hash_, mtime=1000.0, size=10)


class FakeEmbedder(Embedder):
    """Deterministic embedder that always returns the same vector."""

    def __init__(self, vec: list[float], model: str = "fake") -> None:
        self._vec = vec
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return len(self._vec)

    def embed(self, text: str) -> list[float]:
        return list(self._vec)


def _insert_chunk_with_embedding(
    repo: Repository,
    path: Path,
    content: str,
    vec: list[float],
    model: str = "fake",
    hash_suffix: str = "a",
) -> int:
    """Insert file → chunk → embedding; return chunk_id."""
    file_id = repo.upsert_file(_meta(path, hash_=hash_suffix * 64))
    repo.insert_chunks(file_id, [Chunk(path=path, start_line=1, end_line=1, content=content)])
    chunk_id = repo.get_chunks_for_file(file_id)[0].id
    repo.upsert_embedding(chunk_id, vec, model)
    return chunk_id


# ---------------------------------------------------------------------------
# Unit tests: _cosine_top_k_pure
# ---------------------------------------------------------------------------


class TestCosineTopKPure:
    def test_correct_ranking(self) -> None:
        # query aligns perfectly with chunk_id=1
        query = [1.0, 0.0]
        pairs = [
            (1, [1.0, 0.0]),    # cos=1.0
            (2, [0.707, 0.707]),  # cos≈0.707
            (3, [0.0, 1.0]),    # cos=0.0
        ]
        hits = _cosine_top_k_pure(query, pairs, limit=10)
        assert [h.chunk_id for h in hits] == [1, 2, 3]
        assert abs(hits[0].score - 1.0) < 1e-5
        assert abs(hits[2].score - 0.0) < 1e-5

    def test_limit_respected(self) -> None:
        query = [1.0, 0.0]
        pairs = [(i, [float(i), 0.0]) for i in range(1, 11)]
        hits = _cosine_top_k_pure(query, pairs, limit=3)
        assert len(hits) == 3

    def test_zero_query_returns_empty(self) -> None:
        hits = _cosine_top_k_pure([0.0, 0.0], [(1, [1.0, 0.0])], limit=5)
        assert hits == []

    def test_zero_candidate_skipped(self) -> None:
        query = [1.0, 0.0]
        pairs = [(1, [0.0, 0.0]), (2, [1.0, 0.0])]
        hits = _cosine_top_k_pure(query, pairs, limit=5)
        ids = [h.chunk_id for h in hits]
        assert 1 not in ids
        assert 2 in ids

    def test_scores_in_valid_range(self) -> None:
        import random

        random.seed(42)
        query = [random.gauss(0, 1) for _ in range(8)]
        pairs = [(i, [random.gauss(0, 1) for _ in range(8)]) for i in range(20)]
        hits = _cosine_top_k_pure(query, pairs, limit=20)
        for h in hits:
            assert -1.0 - 1e-6 <= h.score <= 1.0 + 1e-6

    def test_sorted_descending(self) -> None:
        query = [1.0, 0.0]
        pairs = [(i, [math.cos(i * 0.3), math.sin(i * 0.3)]) for i in range(10)]
        hits = _cosine_top_k_pure(query, pairs, limit=10)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_empty_pairs(self) -> None:
        assert _cosine_top_k_pure([1.0], [], limit=5) == []

    def test_single_pair(self) -> None:
        hits = _cosine_top_k_pure([1.0, 0.0], [(99, [1.0, 0.0])], limit=1)
        assert len(hits) == 1
        assert hits[0].chunk_id == 99
        assert abs(hits[0].score - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Unit tests: _cosine_top_k_numpy
# ---------------------------------------------------------------------------


class TestCosineTopKNumpy:
    def test_same_ranking_as_pure(self) -> None:
        query = [1.0, 0.0]
        pairs = [
            (1, [1.0, 0.0]),
            (2, [0.707, 0.707]),
            (3, [0.0, 1.0]),
        ]
        numpy_hits = _cosine_top_k_numpy(query, pairs, limit=10)
        pure_hits = _cosine_top_k_pure(query, pairs, limit=10)
        assert [h.chunk_id for h in numpy_hits] == [h.chunk_id for h in pure_hits]
        for n, p in zip(numpy_hits, pure_hits):
            assert abs(n.score - p.score) < 1e-5

    def test_limit_respected(self) -> None:
        query = [1.0, 0.0]
        pairs = [(i, [float(i % 3 + 1), float(i % 2)]) for i in range(20)]
        hits = _cosine_top_k_numpy(query, pairs, limit=5)
        assert len(hits) == 5

    def test_zero_query_returns_empty(self) -> None:
        assert _cosine_top_k_numpy([0.0, 0.0], [(1, [1.0, 0.0])], limit=5) == []

    def test_sorted_descending(self) -> None:
        import random

        random.seed(7)
        query = [random.gauss(0, 1) for _ in range(16)]
        pairs = [(i, [random.gauss(0, 1) for _ in range(16)]) for i in range(30)]
        hits = _cosine_top_k_numpy(query, pairs, limit=30)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Integration: semantic_search end-to-end
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    def test_returns_empty_for_blank_query(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        embedder = FakeEmbedder([1.0, 0.0])
        assert semantic_search("  ", embedder=embedder, repo=repo) == []
        assert semantic_search("", embedder=embedder, repo=repo) == []

    def test_returns_empty_when_no_embeddings(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        # Insert a chunk but no embedding.
        file_id = repo.upsert_file(_meta(Path("/f.py")))
        repo.insert_chunks(file_id, [Chunk(path=Path("/f.py"), start_line=1, end_line=1, content="x")])
        embedder = FakeEmbedder([1.0, 0.0])
        assert semantic_search("hello", embedder=embedder, repo=repo) == []

    def test_correct_ranking(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        # Three chunks with known angle from the query vector [1, 0].
        cid1 = _insert_chunk_with_embedding(repo, Path("/a.py"), "a", [1.0, 0.0], hash_suffix="a")
        cid2 = _insert_chunk_with_embedding(repo, Path("/b.py"), "b", [0.707, 0.707], hash_suffix="b")
        cid3 = _insert_chunk_with_embedding(repo, Path("/c.py"), "c", [0.0, 1.0], hash_suffix="c")

        embedder = FakeEmbedder([1.0, 0.0])
        hits = semantic_search("any query", embedder=embedder, repo=repo)

        assert [h.chunk_id for h in hits] == [cid1, cid2, cid3]
        assert abs(hits[0].score - 1.0) < 1e-5

    def test_limit_respected(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        for i in range(1, 6):
            _insert_chunk_with_embedding(
                repo, Path(f"/{i}.py"), f"chunk {i}", [float(i), 0.0], hash_suffix=chr(96 + i)
            )
        embedder = FakeEmbedder([1.0, 0.0])
        hits = semantic_search("q", embedder=embedder, repo=repo, limit=3)
        assert len(hits) == 3

    def test_filters_by_model(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        # Two chunks: one with "fake" model, one with "other".
        cid_fake = _insert_chunk_with_embedding(repo, Path("/fake.py"), "f", [1.0, 0.0], model="fake")
        _insert_chunk_with_embedding(repo, Path("/other.py"), "o", [1.0, 0.0], model="other", hash_suffix="b")

        embedder = FakeEmbedder([1.0, 0.0], model="fake")
        hits = semantic_search("q", embedder=embedder, repo=repo)
        assert [h.chunk_id for h in hits] == [cid_fake]

    def test_deterministic_output(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        for i in range(1, 4):
            _insert_chunk_with_embedding(
                repo, Path(f"/{i}.py"), f"text {i}", [float(i), float(i % 2)],
                hash_suffix=chr(96 + i),
            )
        embedder = FakeEmbedder([1.0, 0.5])
        hits_a = semantic_search("q", embedder=embedder, repo=repo)
        hits_b = semantic_search("q", embedder=embedder, repo=repo)
        assert hits_a == hits_b

    def test_scores_descending(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        vecs = [[1.0, 0.0], [0.5, 0.866], [0.0, 1.0], [-1.0, 0.0]]
        for i, vec in enumerate(vecs, start=1):
            _insert_chunk_with_embedding(
                repo, Path(f"/{i}.py"), f"c{i}", vec, hash_suffix=chr(96 + i)
            )
        embedder = FakeEmbedder([1.0, 0.0])
        hits = semantic_search("q", embedder=embedder, repo=repo)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_numpy_fallback_to_pure(self, tmp_repo: tuple) -> None:
        """When numpy is unavailable, _cosine_top_k falls back to pure Python."""
        repo, _, _ = tmp_repo
        cid = _insert_chunk_with_embedding(repo, Path("/np.py"), "x", [1.0, 0.0])
        embedder = FakeEmbedder([1.0, 0.0])

        with patch.dict(sys.modules, {"numpy": None}):
            import importlib
            import context_db.retrieval.semantic as sem_mod
            importlib.reload(sem_mod)
            hits = sem_mod.semantic_search("q", embedder=embedder, repo=repo)

        # At least one result returned (pure fallback worked).
        assert len(hits) >= 1


# ---------------------------------------------------------------------------
# Repository helper: get_chunks_by_ids
# ---------------------------------------------------------------------------


class TestGetChunksByIds:
    def test_fetches_correct_chunks(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        p = Path("/src/multi.py")
        file_id = repo.upsert_file(_meta(p))
        chunks = [
            Chunk(path=p, start_line=i, end_line=i, content=f"line {i}")
            for i in range(1, 5)
        ]
        repo.insert_chunks(file_id, chunks)
        all_chunks = repo.get_chunks_for_file(file_id)
        ids = [c.id for c in all_chunks[:2]]

        fetched = repo.get_chunks_by_ids(ids)
        assert {c.id for c in fetched} == set(ids)

    def test_empty_list_returns_empty(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        assert repo.get_chunks_by_ids([]) == []

    def test_unknown_id_returns_empty(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        assert repo.get_chunks_by_ids([999999]) == []
