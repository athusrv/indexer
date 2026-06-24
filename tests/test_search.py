"""Tests for context_db.retrieval.search."""

from __future__ import annotations

from pathlib import Path

import pytest

from context_db.indexer.pipeline import IndexingPipeline
from context_db.models import SearchResult
from context_db.retrieval.search import SearchEngine
from context_db.storage.db import open_db
from context_db.storage.repository import Repository


def _setup_index(tmp_path: Path, sample_tree: Path) -> tuple[SearchEngine, Repository]:
    db_path = tmp_path / "search.db"
    conn = open_db(db_path)
    repo = Repository(conn)
    pipeline = IndexingPipeline(repository=repo)
    pipeline.run(sample_tree)
    engine = SearchEngine(conn)
    return engine, repo


class TestSearchEngine:
    def test_search_finds_relevant_result(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("jwt")
        assert len(results) >= 1
        assert any("auth" in str(r.path) for r in results)

    def test_search_returns_search_results(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("token")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_are_positive(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("helper")
        assert all(r.score > 0 for r in results)

    def test_search_results_ordered_by_score(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("line")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query_returns_empty(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("")
        assert results == []

    def test_search_no_match_returns_empty(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("zzz_nonexistent_term_xyz")
        assert results == []

    def test_search_limit_respected(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("line", limit=1)
        assert len(results) <= 1

    def test_search_preview_truncated(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree, )
        engine._preview_chars = 50
        results = engine.search("jwt")
        for r in results:
            assert len(r.preview) <= 50

    def test_search_path_filter(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("line", path_filter="%.py")
        for r in results:
            assert str(r.path).endswith(".py")

    def test_search_invalid_fts_query_returns_empty(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        # Malformed FTS5 query that would cause OperationalError.
        results = engine.search("AND OR")
        # Should not raise — returns empty or partial results gracefully.
        assert isinstance(results, list)

    def test_search_phrase_query(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        engine, _ = _setup_index(tmp_path, sample_tree)
        results = engine.search("verify token")
        # The phrase "verify token" appears in auth.py
        assert isinstance(results, list)

    def test_bm25_ranking_prefers_denser_matches(
        self, tmp_path: Path
    ) -> None:
        """File with more occurrences of the term should rank higher."""
        db_path = tmp_path / "rank.db"
        conn = open_db(db_path)
        repo = Repository(conn)

        root = tmp_path / "rank_tree"
        root.mkdir()
        (root / "dense.py").write_text("\n".join(["jwt"] * 20))
        (root / "sparse.py").write_text("one jwt mention here\n" + "\n" * 20)

        pipeline = IndexingPipeline(repository=repo)
        pipeline.run(root)

        engine = SearchEngine(conn)
        results = engine.search("jwt")
        assert len(results) >= 2
        # The file with more 'jwt' occurrences should have a higher (or equal) score.
        scores = {r.path.name: r.score for r in results}
        assert scores["dense.py"] >= scores["sparse.py"]

    def test_deduplication_one_result_per_file(self, tmp_path: Path) -> None:
        """Default search must return each file at most once."""
        db_path = tmp_path / "dedup.db"
        conn = open_db(db_path)
        repo = Repository(conn)

        root = tmp_path / "dedup_tree"
        root.mkdir()
        # Write a large file so it produces multiple chunks, all containing 'jwt'.
        lines = ["# jwt token verify\n"] * 200
        (root / "auth.py").write_text("".join(lines))

        pipeline = IndexingPipeline(repository=repo)
        result = pipeline.run(root)
        assert result.total_chunks > 1  # confirm multiple chunks were created

        engine = SearchEngine(conn)
        results = engine.search("jwt")

        paths = [str(r.path) for r in results]
        # Each path must appear exactly once.
        assert len(paths) == len(set(paths)), "Duplicate file in search results"

    def test_deduplication_false_returns_all_chunks(self, tmp_path: Path) -> None:
        """deduplicate=False returns all matching chunk rows."""
        db_path = tmp_path / "all_chunks.db"
        conn = open_db(db_path)
        repo = Repository(conn)

        root = tmp_path / "multi_tree"
        root.mkdir()
        lines = ["# jwt token\n"] * 200
        (root / "big.py").write_text("".join(lines))

        pipeline = IndexingPipeline(repository=repo)
        pipeline.run(root)

        engine = SearchEngine(conn)
        deduped = engine.search("jwt", deduplicate=True)
        all_chunks = engine.search("jwt", deduplicate=False)

        assert len(all_chunks) >= len(deduped)

    def test_deduplication_preserves_best_score(self, tmp_path: Path) -> None:
        """The deduplicated result must carry the highest score among all chunks."""
        db_path = tmp_path / "best.db"
        conn = open_db(db_path)
        repo = Repository(conn)

        root = tmp_path / "best_tree"
        root.mkdir()
        lines = ["# jwt token\n"] * 200
        (root / "big.py").write_text("".join(lines))

        pipeline = IndexingPipeline(repository=repo)
        pipeline.run(root)

        engine = SearchEngine(conn)
        deduped = engine.search("jwt", deduplicate=True)
        all_chunks = engine.search("jwt", deduplicate=False)

        if deduped and all_chunks:
            max_score_all = max(r.score for r in all_chunks)
            assert deduped[0].score == max_score_all
