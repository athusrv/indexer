"""Tests for the embedding step in IndexingPipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from context_db.embeddings.embedder import Embedder
from context_db.indexer.pipeline import IndexingPipeline, RunResult
from context_db.indexer.scanner import Scanner
from context_db.indexer.chunker import Chunker
from context_db.models import Chunk, FileMetadata
from context_db.storage.repository import Repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEmbedder(Embedder):
    """Deterministic embedder that always returns unit vectors of fixed dim."""

    def __init__(self, dims: int = 4, model: str = "fake") -> None:
        self._dims = dims
        self._model = model
        self.embed_calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        return [1.0 / self._dims] * self._dims

    def embed_batch(
        self,
        texts: list[str],
        *,
        progress_callback=None,
    ) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[1.0 / self._dims] * self._dims for _ in texts]


def _write_tree(root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "auth.py").write_text(
        "def verify(token): return True\n", encoding="utf-8"
    )
    (root / "src" / "utils.py").write_text(
        "def helper(x): return x\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Tests: pipeline with embedder=None (default — unchanged behaviour)
# ---------------------------------------------------------------------------


class TestPipelineNoEmbedder:
    def test_run_without_embedder(self, tmp_repo: tuple, tmp_path: Path) -> None:
        repo, conn, _ = tmp_repo
        root = tmp_path / "tree"
        _write_tree(root)
        pipeline = IndexingPipeline(repository=repo)
        result = pipeline.run(root)
        assert result.indexed > 0
        assert result.embedded == 0

    def test_no_embeddings_stored(self, tmp_repo: tuple, tmp_path: Path) -> None:
        repo, conn, _ = tmp_repo
        root = tmp_path / "tree"
        _write_tree(root)
        pipeline = IndexingPipeline(repository=repo)
        pipeline.run(root)
        # No model was specified so nothing should be stored.
        assert repo.get_embedding_count("any-model") == 0


# ---------------------------------------------------------------------------
# Tests: pipeline with embedder (opt-in)
# ---------------------------------------------------------------------------


class TestPipelineWithEmbedder:
    def test_embeds_new_chunks(self, tmp_repo: tuple, tmp_path: Path) -> None:
        repo, conn, _ = tmp_repo
        root = tmp_path / "tree"
        _write_tree(root)

        embedder = FakeEmbedder()
        pipeline = IndexingPipeline(repository=repo, embedder=embedder)
        result = pipeline.run(root)

        assert result.embedded > 0
        assert repo.get_embedding_count(embedder.model_name) == result.embedded

    def test_skips_already_embedded_chunks(self, tmp_repo: tuple, tmp_path: Path) -> None:
        repo, conn, _ = tmp_repo
        root = tmp_path / "tree"
        _write_tree(root)

        embedder = FakeEmbedder()
        pipeline = IndexingPipeline(repository=repo, embedder=embedder)

        # First run: embeds everything.
        result1 = pipeline.run(root)
        assert result1.embedded > 0

        # Second run: no files changed → no new embeddings needed.
        result2 = pipeline.run(root)
        assert result2.embedded == 0
        assert len(embedder.embed_calls) == 1  # only one batch call ever

    def test_reembeds_modified_file(self, tmp_repo: tuple, tmp_path: Path) -> None:
        repo, conn, _ = tmp_repo
        root = tmp_path / "tree"
        _write_tree(root)

        embedder = FakeEmbedder()
        pipeline = IndexingPipeline(repository=repo, embedder=embedder)

        # First run.
        result1 = pipeline.run(root)
        first_count = repo.get_embedding_count(embedder.model_name)

        # Modify a file — its old chunks (and embeddings) are cascade-deleted.
        (root / "src" / "auth.py").write_text(
            "def verify(token): return token is not None\n", encoding="utf-8"
        )
        result2 = pipeline.run(root)
        # At least the modified file's chunks were re-embedded.
        assert result2.embedded > 0

    def test_reembed_flag_clears_all_embeddings(self, tmp_repo: tuple, tmp_path: Path) -> None:
        repo, conn, _ = tmp_repo
        root = tmp_path / "tree"
        _write_tree(root)

        embedder = FakeEmbedder()

        # First run: embed everything.
        pipeline = IndexingPipeline(repository=repo, embedder=embedder, reembed=False)
        result1 = pipeline.run(root)
        count_before = repo.get_embedding_count(embedder.model_name)
        assert count_before > 0

        # Second run with reembed=True: wipe + re-embed.
        pipeline2 = IndexingPipeline(repository=repo, embedder=embedder, reembed=True)
        result2 = pipeline2.run(root)
        count_after = repo.get_embedding_count(embedder.model_name)

        assert result2.embedded == count_before
        assert count_after == count_before

    def test_embed_errors_do_not_crash_pipeline(self, tmp_repo: tuple, tmp_path: Path) -> None:
        """An embedder that raises must not abort the pipeline — embedded=0."""
        repo, conn, _ = tmp_repo
        root = tmp_path / "tree"
        _write_tree(root)

        bad_embedder = FakeEmbedder()
        bad_embedder.embed_batch = MagicMock(side_effect=RuntimeError("model exploded"))  # type: ignore[method-assign]
        bad_embedder._model = None

        pipeline = IndexingPipeline(repository=repo, embedder=bad_embedder)
        result = pipeline.run(root)

        assert result.indexed > 0  # indexing succeeded
        assert result.embedded == 0  # embedding silently failed

    def test_embedded_count_in_run_result(self, tmp_repo: tuple, tmp_path: Path) -> None:
        repo, conn, _ = tmp_repo
        root = tmp_path / "tree"
        _write_tree(root)

        embedder = FakeEmbedder()
        pipeline = IndexingPipeline(repository=repo, embedder=embedder)
        result = pipeline.run(root)

        # embedded == stored count (consistent).
        assert result.embedded == repo.get_embedding_count(embedder.model_name)
