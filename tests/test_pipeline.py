"""Tests for context_db.indexer.pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from context_db.indexer.pipeline import IndexingPipeline, ProgressEvent
from context_db.storage.db import open_db
from context_db.storage.repository import Repository


def _make_pipeline(tmp_path: Path, **kwargs) -> tuple[IndexingPipeline, Repository]:
    db_path = tmp_path / "pipe.db"
    conn = open_db(db_path)
    repo = Repository(conn)
    pipeline = IndexingPipeline(repository=repo, **kwargs)
    return pipeline, repo


class TestIndexingPipeline:
    def test_indexes_files(self, sample_tree: Path, tmp_path: Path) -> None:
        pipeline, repo = _make_pipeline(tmp_path)
        result = pipeline.run(sample_tree)
        assert result.indexed >= 3  # auth.py, utils.py, README.md
        assert result.errors == 0

    def test_does_not_index_ignored_dirs(self, sample_tree: Path, tmp_path: Path) -> None:
        pipeline, repo = _make_pipeline(tmp_path)
        result = pipeline.run(sample_tree)
        all_meta = repo.get_all_file_metadata()
        # Ensure no file whose path has "node_modules" as a directory component.
        assert not any(
            "node_modules" in Path(p).parts for p in all_meta
        )

    def test_incremental_skips_unchanged(self, sample_tree: Path, tmp_path: Path) -> None:
        pipeline, repo = _make_pipeline(tmp_path)
        result1 = pipeline.run(sample_tree)
        assert result1.indexed >= 1

        # Second run: nothing changed.
        result2 = pipeline.run(sample_tree)
        assert result2.indexed == 0  # no new/modified files
        assert result2.skipped == result1.indexed

    def test_incremental_reindexes_modified_file(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        pipeline, repo = _make_pipeline(tmp_path)
        pipeline.run(sample_tree)

        # Modify a file.
        (sample_tree / "src" / "auth.py").write_text(
            "# Modified\ndef new_func(): pass\n",
            encoding="utf-8",
        )
        result2 = pipeline.run(sample_tree)
        assert result2.indexed >= 1

    def test_deleted_file_is_removed(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        f = root / "soon_gone.py"
        f.write_text("x = 1")

        db_path = tmp_path / "test.db"
        conn = open_db(db_path)
        repo = Repository(conn)
        pipeline = IndexingPipeline(repository=repo)

        pipeline.run(root)
        assert len(repo.get_all_file_metadata()) == 1

        # Delete the file.
        f.unlink()
        result = pipeline.run(root)
        assert result.deleted == 1
        assert len(repo.get_all_file_metadata()) == 0

    def test_progress_callback_invoked(self, sample_tree: Path, tmp_path: Path) -> None:
        events: list[ProgressEvent] = []
        pipeline, _ = _make_pipeline(tmp_path, progress_callback=events.append)
        pipeline.run(sample_tree)
        assert len(events) > 0
        assert all(isinstance(e, ProgressEvent) for e in events)

    def test_progress_event_total_ge_current(self, sample_tree: Path, tmp_path: Path) -> None:
        events: list[ProgressEvent] = []
        pipeline, _ = _make_pipeline(tmp_path, progress_callback=events.append)
        pipeline.run(sample_tree)
        for e in events:
            assert e.total >= e.current

    def test_run_returns_run_result(self, sample_tree: Path, tmp_path: Path) -> None:
        from context_db.indexer.pipeline import RunResult
        pipeline, _ = _make_pipeline(tmp_path)
        result = pipeline.run(sample_tree)
        assert isinstance(result, RunResult)
        assert result.duration_s > 0

    def test_total_chunks_reported(self, sample_tree: Path, tmp_path: Path) -> None:
        pipeline, _ = _make_pipeline(tmp_path)
        result = pipeline.run(sample_tree)
        assert result.total_chunks > 0
