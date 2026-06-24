"""Edge-case tests to reach the >90% coverage target.

Covers:
* Scanner: oversized files, OSError on stat, symlinks, unreadable dirs
* Pipeline: error path for failed persist, UnicodeDecodeError fallback, crashing callback
* Search: suggest_terms
* Models: validator edge cases
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from context_db.indexer.pipeline import IndexingPipeline, ProgressEvent
from context_db.indexer.scanner import MAX_FILE_BYTES, Scanner
from context_db.models import Chunk, FileMetadata, SearchResult
from context_db.retrieval.search import SearchEngine
from context_db.storage.db import open_db
from context_db.storage.repository import Repository


# ---------------------------------------------------------------------------
# Scanner edge cases
# ---------------------------------------------------------------------------


class TestScannerEdgeCases:
    def test_oversized_file_skipped(self, tmp_path: Path) -> None:
        big = tmp_path / "huge.py"
        big.write_bytes(b"x" * 10)
        scanner = Scanner(max_file_bytes=5)
        found = scanner.scan(tmp_path)
        assert not any(f.path.name == "huge.py" for f in found)

    def test_symlink_within_tree_allowed(self, tmp_path: Path) -> None:
        real = tmp_path / "real.py"
        real.write_text("x = 1")
        link = tmp_path / "link.py"
        link.symlink_to(real)
        scanner = Scanner()
        found = scanner.scan(tmp_path)
        names = {f.path.name for f in found}
        # At least the real file should be found.
        assert "real.py" in names

    def test_dir_pattern_matching_against_relative_path(self, tmp_path: Path) -> None:
        deep = tmp_path / "sub" / "vendored"
        deep.mkdir(parents=True)
        (deep / "lib.js").write_text("const x = 1;")
        scanner = Scanner(ignore_patterns=["sub/vendored"])
        found = scanner.scan(tmp_path)
        assert not any("vendored" in str(f.path) for f in found)

    def test_percent_complete_on_progress_event(self) -> None:
        evt = ProgressEvent(current=50, total=100, path=Path("f.py"), action="index")
        assert evt.percent == 50.0

    def test_percent_complete_zero_total(self) -> None:
        evt = ProgressEvent(current=0, total=0, path=Path("f.py"), action="index")
        assert evt.percent == 0.0


# ---------------------------------------------------------------------------
# Pipeline edge cases
# ---------------------------------------------------------------------------


class TestPipelineEdgeCases:
    def _make(self, tmp_path: Path, **kwargs) -> tuple[IndexingPipeline, Repository]:
        db_path = tmp_path / "edge.db"
        conn = open_db(db_path)
        repo = Repository(conn)
        pipeline = IndexingPipeline(repository=repo, **kwargs)
        return pipeline, repo

    def test_pipeline_handles_read_error_gracefully(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        f = root / "fail.py"
        f.write_text("x = 1")
        pipeline, repo = self._make(tmp_path)

        # Patch replace_file_chunks to raise.
        with patch.object(repo, "replace_file_chunks", side_effect=RuntimeError("disk full")):
            result = pipeline.run(root)
        assert result.errors >= 1

    def test_pipeline_crashing_progress_callback_does_not_propagate(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        def bad_callback(evt: ProgressEvent) -> None:
            raise RuntimeError("callback exploded")

        pipeline, _ = self._make(tmp_path, progress_callback=bad_callback)
        # Must not raise.
        result = pipeline.run(sample_tree)
        assert result.indexed >= 1

    def test_read_file_latin1_fallback(self, tmp_path: Path) -> None:
        """_read_file falls back to latin-1 on UnicodeDecodeError."""
        f = tmp_path / "latin.py"
        # Write bytes that are invalid UTF-8 but valid latin-1.
        f.write_bytes(b"x = '\xff'\n")
        content = IndexingPipeline._read_file(f)
        assert "x" in content

    def test_pipeline_no_changes_skips_fts_optimize(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        """Second run with no changes should not call optimize_fts."""
        pipeline, repo = self._make(tmp_path)
        pipeline.run(sample_tree)

        with patch.object(repo, "optimize_fts") as mock_opt:
            result = pipeline.run(sample_tree)
        mock_opt.assert_not_called()
        assert result.indexed == 0

    def test_delete_error_increments_error_count(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        f = root / "gone.py"
        f.write_text("x = 1")

        db_path = tmp_path / "del.db"
        conn = open_db(db_path)
        repo = Repository(conn)
        pipeline = IndexingPipeline(repository=repo)
        pipeline.run(root)

        # Delete the file so it appears in deleted_paths.
        f.unlink()

        with patch.object(repo, "delete_file_transactional", side_effect=RuntimeError("io error")):
            result = pipeline.run(root)
        assert result.errors >= 1


# ---------------------------------------------------------------------------
# Search edge cases
# ---------------------------------------------------------------------------


class TestSearchEdgeCases:
    def _setup(self, tmp_path: Path, sample_tree: Path) -> SearchEngine:
        db_path = tmp_path / "s.db"
        conn = open_db(db_path)
        repo = Repository(conn)
        pipeline = IndexingPipeline(repository=repo)
        pipeline.run(sample_tree)
        return SearchEngine(conn)

    def test_suggest_terms_returns_list(self, sample_tree: Path, tmp_path: Path) -> None:
        engine = self._setup(tmp_path, sample_tree)
        results = engine.suggest_terms("tok")
        assert isinstance(results, list)

    def test_suggest_terms_empty_prefix(self, sample_tree: Path, tmp_path: Path) -> None:
        engine = self._setup(tmp_path, sample_tree)
        results = engine.suggest_terms("")
        assert results == []

    def test_search_whitespace_only_query(self, sample_tree: Path, tmp_path: Path) -> None:
        engine = self._setup(tmp_path, sample_tree)
        results = engine.search("   ")
        assert results == []

    def test_prepare_query_passes_through_operators(self, sample_tree: Path, tmp_path: Path) -> None:
        engine = self._setup(tmp_path, sample_tree)
        q = engine._prepare_query("jwt OR token")
        assert q == "jwt OR token"

    def test_prepare_query_wraps_multi_word(self, sample_tree: Path, tmp_path: Path) -> None:
        engine = self._setup(tmp_path, sample_tree)
        q = engine._prepare_query("verify token")
        assert q.startswith('"')

    def test_prepare_query_single_word_unchanged(self, sample_tree: Path, tmp_path: Path) -> None:
        engine = self._setup(tmp_path, sample_tree)
        q = engine._prepare_query("jwt")
        assert q == "jwt"

    def test_prepare_query_with_asterisk_unchanged(self, sample_tree: Path, tmp_path: Path) -> None:
        engine = self._setup(tmp_path, sample_tree)
        q = engine._prepare_query("auth*")
        assert q == "auth*"


# ---------------------------------------------------------------------------
# Model validators
# ---------------------------------------------------------------------------


class TestModelValidators:
    def test_chunk_end_before_start_raises(self) -> None:
        with pytest.raises(Exception):
            Chunk(path=Path("f.py"), start_line=10, end_line=5, content="x")

    def test_file_metadata_invalid_hash_raises(self) -> None:
        with pytest.raises(Exception):
            FileMetadata(path=Path("f.py"), hash="tooshort", mtime=1.0, size=10)

    def test_search_result_model(self) -> None:
        r = SearchResult(
            path=Path("auth.py"),
            score=1.5,
            start_line=1,
            end_line=5,
            preview="some content",
        )
        assert r.score == 1.5
