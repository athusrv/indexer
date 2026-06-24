"""Tests for context_db.retrieval.hybrid."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from context_db.embeddings.embedder import Embedder
from context_db.models import Chunk, FileMetadata, HybridResult
from context_db.retrieval.hybrid import (
    _LexicalHit,
    _fts_chunk_search,
    _normalize,
    _prepare_fts_query,
    hybrid_search,
)
from context_db.storage.repository import Repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(path: Path, hash_suffix: str = "a") -> FileMetadata:
    return FileMetadata(path=path, hash=hash_suffix * 64, mtime=1000.0, size=10)


class FakeEmbedder(Embedder):
    """Returns a fixed vector for every query."""

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


def _index_file(
    repo: Repository,
    path: Path,
    content: str,
    hash_suffix: str = "a",
) -> int:
    """Insert file + one chunk via replace_file_chunks; return chunk_id."""
    meta = _meta(path, hash_suffix=hash_suffix)
    chunk = Chunk(path=path, start_line=1, end_line=1, content=content)
    repo.replace_file_chunks(meta, [chunk])
    file = repo.get_file_by_path(path)
    return repo.get_chunks_for_file(file.id)[0].id


# ---------------------------------------------------------------------------
# Unit: _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_empty_input(self) -> None:
        assert _normalize({}) == {}

    def test_single_entry(self) -> None:
        result = _normalize({1: 5.0})
        assert result == {1: 1.0}

    def test_all_same_values(self) -> None:
        result = _normalize({1: 3.0, 2: 3.0, 3: 3.0})
        assert all(v == 1.0 for v in result.values())

    def test_two_values(self) -> None:
        result = _normalize({1: 0.0, 2: 1.0})
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_general_case(self) -> None:
        result = _normalize({1: 10.0, 2: 20.0, 3: 30.0})
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.5)
        assert result[3] == pytest.approx(1.0)

    def test_output_range(self) -> None:
        import random
        random.seed(99)
        scores = {i: random.uniform(-5, 5) for i in range(20)}
        result = _normalize(scores)
        for v in result.values():
            assert 0.0 <= v <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Unit: _prepare_fts_query
# ---------------------------------------------------------------------------


class TestPrepareFtsQuery:
    def test_single_word_unchanged(self) -> None:
        assert _prepare_fts_query("jwt") == "jwt"

    def test_multi_word_quoted(self) -> None:
        assert _prepare_fts_query("jwt token") == '"jwt token"'

    def test_fts_operators_pass_through(self) -> None:
        assert _prepare_fts_query("jwt AND token") == "jwt AND token"

    def test_quoted_query_unchanged(self) -> None:
        assert _prepare_fts_query('"exact phrase"') == '"exact phrase"'

    def test_star_wildcard_unchanged(self) -> None:
        assert _prepare_fts_query("auth*") == "auth*"


# ---------------------------------------------------------------------------
# Unit: _fts_chunk_search
# ---------------------------------------------------------------------------


class TestFtsChunkSearch:
    def test_finds_matching_content(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        _index_file(repo, Path("/auth.py"), "jwt token verify authentication")
        hits = _fts_chunk_search("jwt", conn, limit=10)
        assert len(hits) == 1
        assert hits[0].chunk_id > 0
        assert hits[0].path == Path("/auth.py")

    def test_no_match_returns_empty(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        _index_file(repo, Path("/auth.py"), "jwt token")
        hits = _fts_chunk_search("xyzzy_nomatch_abc123", conn, limit=10)
        assert hits == []

    def test_limit_respected(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        for i in range(5):
            _index_file(
                repo, Path(f"/{i}.py"), f"authentication token {i}", hash_suffix=chr(97 + i)
            )
        hits = _fts_chunk_search("authentication", conn, limit=3)
        assert len(hits) <= 3

    def test_scores_positive(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        _index_file(repo, Path("/f.py"), "hello world")
        hits = _fts_chunk_search("hello", conn, limit=5)
        assert all(h.score > 0 for h in hits)

    def test_invalid_fts_query_returns_empty(self, tmp_repo: tuple) -> None:
        _, conn, _ = tmp_repo
        # Malformed FTS5 expression should not raise.
        hits = _fts_chunk_search('AND OR """', conn, limit=5)
        assert hits == []


# ---------------------------------------------------------------------------
# Integration: hybrid_search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    # ── Empty / trivial ───────────────────────────────────────────────────

    def test_empty_query_returns_empty(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        embedder = FakeEmbedder([1.0, 0.0])
        assert hybrid_search("", embedder=embedder, repo=repo, conn=conn) == []
        assert hybrid_search("  ", embedder=embedder, repo=repo, conn=conn) == []

    def test_no_indexed_data_returns_empty(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        embedder = FakeEmbedder([1.0, 0.0])
        assert hybrid_search("query", embedder=embedder, repo=repo, conn=conn) == []

    # ── Match type ────────────────────────────────────────────────────────

    def test_lexical_only_match_type(self, tmp_repo: tuple) -> None:
        """Chunks in FTS but NO embeddings → match_type='lexical'."""
        repo, conn, _ = tmp_repo
        _index_file(repo, Path("/auth.py"), "jwt token verify")
        # embedder model name won't match any stored embedding (none stored)
        embedder = FakeEmbedder([1.0, 0.0], model="no-embeddings-model")
        results = hybrid_search("jwt", embedder=embedder, repo=repo, conn=conn)
        assert len(results) > 0
        assert all(r.match_type == "lexical" for r in results)

    def test_semantic_only_match_type(self, tmp_repo: tuple) -> None:
        """Chunk has an embedding but FTS query returns nothing → 'semantic'."""
        repo, conn, _ = tmp_repo
        # Index chunk with content that won't match a gibberish query.
        chunk_id = _index_file(repo, Path("/sem.py"), "hello world", hash_suffix="b")
        repo.upsert_embedding(chunk_id, [1.0, 0.0], "fake")

        embedder = FakeEmbedder([1.0, 0.0], model="fake")
        # Query that FTS will not find.
        results = hybrid_search("xyzzy_nomatch_abc123", embedder=embedder, repo=repo, conn=conn)
        assert len(results) > 0
        assert all(r.match_type == "semantic" for r in results)

    def test_hybrid_match_type(self, tmp_repo: tuple) -> None:
        """Chunk found by both FTS and semantic → 'hybrid'."""
        repo, conn, _ = tmp_repo
        chunk_id = _index_file(repo, Path("/both.py"), "jwt token verify")
        repo.upsert_embedding(chunk_id, [1.0, 0.0], "fake")

        embedder = FakeEmbedder([1.0, 0.0], model="fake")
        results = hybrid_search("jwt", embedder=embedder, repo=repo, conn=conn)
        hybrid_results = [r for r in results if r.match_type == "hybrid"]
        assert len(hybrid_results) >= 1

    # ── Scoring ───────────────────────────────────────────────────────────

    def test_scores_descending(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        for i in range(1, 5):
            cid = _index_file(
                repo, Path(f"/{i}.py"), f"authentication token {i}", hash_suffix=chr(96 + i)
            )
            repo.upsert_embedding(cid, [float(i), 0.0], "fake")

        embedder = FakeEmbedder([1.0, 0.0], model="fake")
        results = hybrid_search("authentication", embedder=embedder, repo=repo, conn=conn)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_between_zero_and_one(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        cid = _index_file(repo, Path("/f.py"), "hello world")
        repo.upsert_embedding(cid, [1.0, 0.0], "fake")
        embedder = FakeEmbedder([1.0, 0.0], model="fake")
        results = hybrid_search("hello", embedder=embedder, repo=repo, conn=conn)
        for r in results:
            assert 0.0 <= r.score <= 1.0 + 1e-9

    def test_score_formula_lexical_only(self, tmp_repo: tuple) -> None:
        """When only lexical signal is present the max score is LEXICAL_WEIGHT."""
        from context_db.retrieval.hybrid import _LEXICAL_WEIGHT

        repo, conn, _ = tmp_repo
        _index_file(repo, Path("/a.py"), "jwt verify token")
        embedder = FakeEmbedder([1.0, 0.0], model="unused-model")

        results = hybrid_search("jwt", embedder=embedder, repo=repo, conn=conn)
        assert len(results) == 1
        # Only one lexical hit → lex_norm=1.0, sem_norm=0.0
        assert results[0].score == pytest.approx(_LEXICAL_WEIGHT, abs=1e-6)

    def test_score_formula_semantic_only(self, tmp_repo: tuple) -> None:
        """When only semantic signal is present the max score is SEMANTIC_WEIGHT."""
        from context_db.retrieval.hybrid import _SEMANTIC_WEIGHT

        repo, conn, _ = tmp_repo
        chunk_id = _index_file(repo, Path("/s.py"), "hello world", hash_suffix="b")
        repo.upsert_embedding(chunk_id, [1.0, 0.0], "fake")
        embedder = FakeEmbedder([1.0, 0.0], model="fake")

        results = hybrid_search("xyzzy_nomatch_abc123", embedder=embedder, repo=repo, conn=conn)
        assert len(results) == 1
        assert results[0].score == pytest.approx(_SEMANTIC_WEIGHT, abs=1e-6)

    # ── Deduplication ─────────────────────────────────────────────────────

    def test_chunk_appears_once_when_in_both_sets(self, tmp_repo: tuple) -> None:
        """A chunk found by both FTS and semantic must appear exactly once."""
        repo, conn, _ = tmp_repo
        chunk_id = _index_file(repo, Path("/dup.py"), "jwt authentication token")
        repo.upsert_embedding(chunk_id, [1.0, 0.0], "fake")
        embedder = FakeEmbedder([1.0, 0.0], model="fake")

        results = hybrid_search("jwt", embedder=embedder, repo=repo, conn=conn)
        paths = [str(r.path) for r in results]
        assert len(paths) == len(set(paths))

    # ── Limit ─────────────────────────────────────────────────────────────

    def test_limit_respected(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        for i in range(1, 8):
            cid = _index_file(
                repo, Path(f"/{i}.py"), f"jwt authentication {i}", hash_suffix=chr(96 + i)
            )
            repo.upsert_embedding(cid, [float(i), 0.0], "fake")
        embedder = FakeEmbedder([1.0, 0.0], model="fake")
        results = hybrid_search("jwt", embedder=embedder, repo=repo, conn=conn, limit=3)
        assert len(results) <= 3

    # ── Output shape ──────────────────────────────────────────────────────

    def test_result_fields_populated(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        cid = _index_file(repo, Path("/shape.py"), "jwt verify token")
        repo.upsert_embedding(cid, [1.0, 0.0], "fake")
        embedder = FakeEmbedder([1.0, 0.0], model="fake")

        results = hybrid_search("jwt", embedder=embedder, repo=repo, conn=conn)
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, HybridResult)
        assert r.path == Path("/shape.py")
        assert r.score > 0
        assert r.match_type in {"lexical", "semantic", "hybrid"}
        assert r.start_line >= 1
        assert r.end_line >= r.start_line
        assert len(r.preview) > 0

    def test_preview_truncated_to_preview_chars(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        long_content = "jwt " + "x" * 1000
        _index_file(repo, Path("/long.py"), long_content)
        embedder = FakeEmbedder([1.0], model="unused")
        results = hybrid_search("jwt", embedder=embedder, repo=repo, conn=conn, preview_chars=50)
        assert all(len(r.preview) <= 50 for r in results)

    # ── Multi-chunk file ──────────────────────────────────────────────────

    def test_multiple_chunks_per_file_can_appear(self, tmp_repo: tuple) -> None:
        """hybrid_search is chunk-level, not file-level — multiple chunks ok."""
        repo, conn, _ = tmp_repo
        p = Path("/multi.py")
        meta = _meta(p, hash_suffix="z")
        chunks = [
            Chunk(path=p, start_line=1, end_line=1, content="jwt authentication"),
            Chunk(path=p, start_line=2, end_line=2, content="jwt token verify"),
        ]
        repo.replace_file_chunks(meta, chunks)
        embedder = FakeEmbedder([1.0], model="unused")
        results = hybrid_search("jwt", embedder=embedder, repo=repo, conn=conn)
        # At least 2 chunk-level results from the same file.
        assert len(results) >= 2

    # ── Repository: get_chunk_file_info ───────────────────────────────────

    def test_get_chunk_file_info_correct(self, tmp_repo: tuple) -> None:
        repo, conn, _ = tmp_repo
        cid = _index_file(repo, Path("/info.py"), "some content here")
        infos = repo.get_chunk_file_info([cid])
        assert len(infos) == 1
        assert infos[0].chunk_id == cid
        assert infos[0].path == Path("/info.py")
        assert infos[0].start_line == 1

    def test_get_chunk_file_info_empty(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        assert repo.get_chunk_file_info([]) == []

    def test_get_chunk_file_info_unknown_id(self, tmp_repo: tuple) -> None:
        repo, _, _ = tmp_repo
        assert repo.get_chunk_file_info([999999]) == []
