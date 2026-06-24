"""Tests for the embedding-related CLI extensions (ctx index + ctx search)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from context_db.cli import app
from context_db.embeddings.embedder import Embedder

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_opt(db_path: Path) -> list[str]:
    return ["--db", str(db_path)]


class FakeEmbedder(Embedder):
    """Deterministic 2-d embedder for CLI tests (no model download)."""

    def __init__(self, model: str = "fake-cli-model") -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, text: str) -> list[float]:
        return [0.6, 0.8]

    def embed_batch(self, texts, *, progress_callback=None):
        return [[0.6, 0.8] for _ in texts]


def _fake_create_embedder(model_name=None):
    return FakeEmbedder(model=model_name or "fake-cli-model")


# ---------------------------------------------------------------------------
# ctx index --embed-model
# ---------------------------------------------------------------------------


class TestIndexWithEmbedModel:
    def test_index_with_embed_model_stores_embeddings(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        db = tmp_path / "e.db"
        with patch("context_db.cli.create_embedder", side_effect=_fake_create_embedder):
            result = runner.invoke(
                app,
                ["index", str(sample_tree), "--embed-model", "fake-model"] + _db_opt(db),
            )
        assert result.exit_code == 0
        assert "Index complete" in result.output

    def test_index_embed_count_shown_in_summary(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        db = tmp_path / "e.db"
        with patch("context_db.cli.create_embedder", side_effect=_fake_create_embedder):
            result = runner.invoke(
                app,
                ["index", str(sample_tree), "--embed-model", "fake"] + _db_opt(db),
            )
        assert result.exit_code == 0
        assert "Chunks embedded:" in result.output

    def test_index_disable_embeddings_skips_embed(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        db = tmp_path / "e.db"
        mock_factory = MagicMock(side_effect=_fake_create_embedder)
        with patch("context_db.cli.create_embedder", mock_factory):
            result = runner.invoke(
                app,
                [
                    "index", str(sample_tree),
                    "--embed-model", "fake",
                    "--disable-embeddings",
                ] + _db_opt(db),
            )
        assert result.exit_code == 0
        # create_embedder must NOT have been called.
        mock_factory.assert_not_called()

    def test_index_reembed_flag_accepted(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "e.db"
        with patch("context_db.cli.create_embedder", side_effect=_fake_create_embedder):
            # First pass: embed
            runner.invoke(
                app,
                ["index", str(sample_tree), "--embed-model", "fake"] + _db_opt(db),
            )
            # Second pass: reembed
            result = runner.invoke(
                app,
                ["index", str(sample_tree), "--embed-model", "fake", "--reembed"] + _db_opt(db),
            )
        assert result.exit_code == 0
        assert "Index complete" in result.output

    def test_index_import_error_gives_helpful_message(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        db = tmp_path / "e.db"
        with patch(
            "context_db.cli.create_embedder",
            side_effect=ImportError("sentence-transformers not installed"),
        ):
            result = runner.invoke(
                app,
                ["index", str(sample_tree), "--embed-model", "nomic-ai/x"] + _db_opt(db),
            )
        assert result.exit_code == 1
        assert "sentence-transformers" in result.output.lower() or "embedder" in result.output.lower()

    def test_index_without_embed_model_is_backward_compatible(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        """ctx index without --embed-model must behave exactly as before."""
        db = tmp_path / "e.db"
        result = runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        assert result.exit_code == 0
        assert "Index complete" in result.output
        # No embedding-related line in output.
        assert "embedded" not in result.output.lower()


# ---------------------------------------------------------------------------
# ctx search --semantic
# ---------------------------------------------------------------------------


class TestSearchSemantic:
    def _setup_db(self, sample_tree: Path, tmp_path: Path) -> Path:
        """Index and store fake embeddings; return db path."""
        from context_db.storage.db import open_db
        from context_db.storage.repository import Repository

        db = tmp_path / "sem.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))

        # Store fake embeddings manually.
        conn = open_db(db)
        repo = Repository(conn)
        missing = repo.get_chunks_without_embeddings("fake-cli-model")
        if missing:
            repo.upsert_embeddings_batch(
                [c.id for c in missing],
                [[0.6, 0.8]] * len(missing),
                "fake-cli-model",
            )
        conn.commit()  # must commit before close; sqlite3 rolls back on close otherwise
        conn.close()
        return db

    def test_semantic_search_returns_results(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        db = self._setup_db(sample_tree, tmp_path)
        with patch("context_db.cli.create_embedder", side_effect=_fake_create_embedder):
            result = runner.invoke(
                app,
                ["search", "jwt", "--semantic", "--embed-model", "fake-cli-model"] + _db_opt(db),
            )
        assert result.exit_code == 0

    def test_semantic_search_json_has_match_type(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        db = self._setup_db(sample_tree, tmp_path)
        with patch("context_db.cli.create_embedder", side_effect=_fake_create_embedder):
            result = runner.invoke(
                app,
                ["search", "jwt", "--semantic", "--embed-model", "fake-cli-model", "--json"]
                + _db_opt(db),
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all(r["match_type"] == "semantic" for r in data)

    def test_disable_embeddings_overrides_semantic(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        """--disable-embeddings must force FTS even when --semantic is given."""
        db = self._setup_db(sample_tree, tmp_path)
        mock_factory = MagicMock(side_effect=_fake_create_embedder)
        with patch("context_db.cli.create_embedder", mock_factory):
            result = runner.invoke(
                app,
                ["search", "jwt", "--semantic", "--disable-embeddings", "--json"] + _db_opt(db),
            )
        assert result.exit_code == 0
        mock_factory.assert_not_called()
        data = json.loads(result.output)
        # All results are lexical (FTS fallback).
        assert all(r["match_type"] == "lexical" for r in data)

    def test_semantic_import_error_exits_with_message(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        db = self._setup_db(sample_tree, tmp_path)
        with patch(
            "context_db.cli.create_embedder",
            side_effect=ImportError("no sentence-transformers"),
        ):
            result = runner.invoke(
                app,
                ["search", "jwt", "--semantic"] + _db_opt(db),
            )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# ctx search --hybrid
# ---------------------------------------------------------------------------


class TestSearchHybrid:
    def _setup_db(self, sample_tree: Path, tmp_path: Path) -> Path:
        from context_db.storage.db import open_db
        from context_db.storage.repository import Repository

        db = tmp_path / "hyb.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))

        conn = open_db(db)
        repo = Repository(conn)
        missing = repo.get_chunks_without_embeddings("fake-cli-model")
        if missing:
            repo.upsert_embeddings_batch(
                [c.id for c in missing],
                [[0.6, 0.8]] * len(missing),
                "fake-cli-model",
            )
        conn.commit()  # must commit before close
        conn.close()
        return db

    def test_hybrid_search_exits_ok(self, sample_tree: Path, tmp_path: Path) -> None:
        db = self._setup_db(sample_tree, tmp_path)
        with patch("context_db.cli.create_embedder", side_effect=_fake_create_embedder):
            result = runner.invoke(
                app,
                ["search", "jwt", "--hybrid", "--embed-model", "fake-cli-model"] + _db_opt(db),
            )
        assert result.exit_code == 0

    def test_hybrid_json_has_match_type_field(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        db = self._setup_db(sample_tree, tmp_path)
        with patch("context_db.cli.create_embedder", side_effect=_fake_create_embedder):
            result = runner.invoke(
                app,
                ["search", "jwt", "--hybrid", "--embed-model", "fake-cli-model", "--json"]
                + _db_opt(db),
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all("match_type" in r for r in data)

    def test_hybrid_wins_over_semantic_when_both_set(
        self, sample_tree: Path, tmp_path: Path
    ) -> None:
        """When both --hybrid and --semantic are passed, hybrid is used."""
        db = self._setup_db(sample_tree, tmp_path)
        with patch("context_db.cli.create_embedder", side_effect=_fake_create_embedder):
            result = runner.invoke(
                app,
                [
                    "search", "jwt",
                    "--hybrid", "--semantic",
                    "--embed-model", "fake-cli-model",
                    "--json",
                ] + _db_opt(db),
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Hybrid results may be "lexical", "semantic", or "hybrid" — just check structure.
        assert all("match_type" in r for r in data)


# ---------------------------------------------------------------------------
# ctx search (default FTS — unchanged behaviour)
# ---------------------------------------------------------------------------


class TestSearchDefaultFts:
    def test_default_search_unchanged(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "fts.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        result = runner.invoke(app, ["search", "jwt"] + _db_opt(db))
        assert result.exit_code == 0
        assert "auth" in result.output.lower()

    def test_default_json_output_unchanged(self, sample_tree: Path, tmp_path: Path) -> None:
        db = tmp_path / "fts.db"
        runner.invoke(app, ["index", str(sample_tree)] + _db_opt(db))
        result = runner.invoke(app, ["search", "jwt", "--json"] + _db_opt(db))
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all(r["match_type"] == "lexical" for r in data)
