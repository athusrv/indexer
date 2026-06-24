"""Tests for context_db.cli (Typer commands)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from context_db.cli import app

runner = CliRunner()


def _db_opt(db_path: Path) -> list[str]:
    return ["--db", str(db_path)]


class TestInit:
    def test_init_creates_database(self, tmp_path: Path) -> None:
        db = tmp_path / "ctx.db"
        result = runner.invoke(app, ["init"] + _db_opt(db))
        assert result.exit_code == 0
        assert db.exists()

    def test_init_already_exists(self, tmp_path: Path) -> None:
        db = tmp_path / "ctx.db"
        db.touch()
        result = runner.invoke(app, ["init"] + _db_opt(db))
        assert result.exit_code == 0
        assert "already exists" in result.output

    def test_init_uses_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db = tmp_path / "env.db"
        monkeypatch.setenv("CTX_DB", str(db))
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert db.exists()


class TestIndex:
    def test_index_directory(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        result = runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        assert result.exit_code == 0
        assert "Index complete" in result.output

    def test_index_shows_file_count(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        result = runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        assert "Files indexed:" in result.output

    def test_index_incremental_skips(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        result = runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        assert result.exit_code == 0
        assert "Files skipped:" in result.output

    def test_index_verbose_flag(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        result = runner.invoke(app, ["index", str(sample_tree), "-v"] + _db_opt(db))
        assert result.exit_code == 0

    def test_index_custom_chunk_chars(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        result = runner.invoke(
            app,
            ["index", str(sample_tree), "--chunk-chars", "500"] + _db_opt(db),
        )
        assert result.exit_code == 0

    def test_index_with_ignore_pattern(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "idx.db"
        result = runner.invoke(
            app,
            ["index", str(sample_tree), "--ignore", "*.md"] + _db_opt(db),
        )
        assert result.exit_code == 0


class TestSearch:
    def _indexed_db(self, sample_tree: Path, tmp_path: Path) -> Path:
        db = tmp_path / "search.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        return db

    def test_search_finds_results(self, sample_tree: Path, tmp_path: Path) -> None:
        db = self._indexed_db(sample_tree, tmp_path)
        result = runner.invoke(app, ["search", "jwt"] + _db_opt(db))
        assert result.exit_code == 0
        assert "auth" in result.output.lower()

    def test_search_no_results(self, sample_tree: Path, tmp_path: Path) -> None:
        db = self._indexed_db(sample_tree, tmp_path)
        result = runner.invoke(app, ["search", "zzz_no_match_xyz"] + _db_opt(db))
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_json_output(self, sample_tree: Path, tmp_path: Path) -> None:
        import json
        db = self._indexed_db(sample_tree, tmp_path)
        result = runner.invoke(app, ["search", "jwt", "--json"] + _db_opt(db))
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_search_with_limit(self, sample_tree: Path, tmp_path: Path) -> None:
        db = self._indexed_db(sample_tree, tmp_path)
        result = runner.invoke(app, ["search", "line", "-n", "1"] + _db_opt(db))
        assert result.exit_code == 0

    def test_search_with_path_filter(self, sample_tree: Path, tmp_path: Path) -> None:
        db = self._indexed_db(sample_tree, tmp_path)
        result = runner.invoke(app, ["search", "line", "--path", "%.py"] + _db_opt(db))
        assert result.exit_code == 0

    def test_search_no_db(self, tmp_path: Path) -> None:
        db = tmp_path / "missing.db"
        result = runner.invoke(app, ["search", "jwt"] + _db_opt(db))
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_search_all_chunks_flag(self, sample_tree: Path, tmp_path: Path) -> None:
        db = self._indexed_db(sample_tree, tmp_path)
        result = runner.invoke(app, ["search", "line", "--all-chunks"] + _db_opt(db))
        assert result.exit_code == 0

    def test_search_default_deduplicates(self, tmp_path: Path) -> None:
        """Each file must appear at most once in default search output."""
        import json

        root = tmp_path / "root"
        root.mkdir()
        # A large file that generates multiple chunks, all containing 'jwt'.
        (root / "auth.py").write_text("# jwt token verify\n" * 200)

        db = tmp_path / "dd.db"
        runner.invoke(app, ["index", str(root)] + _db_opt(db))

        result = runner.invoke(app, ["search", "jwt", "--json"] + _db_opt(db))
        assert result.exit_code == 0
        data = json.loads(result.output)
        paths = [r["path"] for r in data]
        assert len(paths) == len(set(paths)), "Duplicate file in CLI search output"


class TestStats:
    def test_stats_shows_counts(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "stats.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        result = runner.invoke(app, ["stats"] + _db_opt(db))
        assert result.exit_code == 0
        assert "Files indexed:" in result.output

    def test_stats_no_db(self, tmp_path: Path) -> None:
        db = tmp_path / "missing.db"
        result = runner.invoke(app, ["stats"] + _db_opt(db))
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestReset:
    def test_reset_clears_data(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "reset.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        result = runner.invoke(app, ["reset", "--yes"] + _db_opt(db))
        assert result.exit_code == 0
        assert "reset" in result.output.lower()

    def test_reset_no_db(self, tmp_path: Path) -> None:
        db = tmp_path / "missing.db"
        result = runner.invoke(app, ["reset", "--yes"] + _db_opt(db))
        assert result.exit_code == 1

    def test_reset_confirmation_aborted(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "reset.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        # Provide "n" to decline the confirmation prompt.
        result = runner.invoke(app, ["reset"] + _db_opt(db), input="n\n")
        assert result.exit_code != 0
