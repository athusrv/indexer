"""Tests for the embeddings subsystem (Embedder ABC + LocalEmbedder).

All tests mock sentence-transformers so no model download is required.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from context_db.embeddings.embedder import DEFAULT_MODEL, Embedder, create_embedder


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_st(dims: int = 4) -> MagicMock:
    """Return a mock SentenceTransformer that produces deterministic vectors."""
    import numpy as np

    st = MagicMock()

    def _encode(texts, convert_to_numpy=False):
        return np.ones((len(texts), dims), dtype="float32") * 0.5

    st.encode.side_effect = _encode
    return st


# ---------------------------------------------------------------------------
# Embedder ABC
# ---------------------------------------------------------------------------


class ConcreteEmbedder(Embedder):
    """Minimal concrete implementation for testing the base-class default behaviour."""

    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def dimensions(self) -> int:
        return 3

    def embed(self, text: str) -> list[float]:
        return [float(len(text))] * 3


class TestEmbedderBase:
    def test_embed_returns_list_of_floats(self) -> None:
        e = ConcreteEmbedder()
        vec = e.embed("hello")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch_default_delegates_to_embed(self) -> None:
        e = ConcreteEmbedder()
        texts = ["a", "bb", "ccc"]
        results = e.embed_batch(texts)
        assert len(results) == 3
        for text, vec in zip(texts, results):
            assert vec == e.embed(text)

    def test_embed_batch_progress_callback(self) -> None:
        e = ConcreteEmbedder()
        calls: list[tuple[int, int]] = []
        e.embed_batch(["x", "y", "z"], progress_callback=lambda d, t: calls.append((d, t)))
        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_embed_batch_empty_returns_empty(self) -> None:
        e = ConcreteEmbedder()
        assert e.embed_batch([]) == []


# ---------------------------------------------------------------------------
# create_embedder factory
# ---------------------------------------------------------------------------


class TestCreateEmbedder:
    def test_returns_local_embedder_instance(self) -> None:
        from context_db.embeddings.providers.local import LocalEmbedder

        e = create_embedder()
        assert isinstance(e, LocalEmbedder)

    def test_uses_default_model(self) -> None:
        e = create_embedder()
        assert e.model_name == DEFAULT_MODEL

    def test_accepts_custom_model(self) -> None:
        e = create_embedder("BAAI/bge-small-en-v1.5")
        assert e.model_name == "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# LocalEmbedder — model loading
# ---------------------------------------------------------------------------


class TestLocalEmbedderLoading:
    def test_model_loaded_lazily(self) -> None:
        from context_db.embeddings.providers.local import LocalEmbedder

        e = LocalEmbedder()
        assert e._model is None  # not loaded yet

    def test_raises_import_error_when_st_missing(self) -> None:
        """Simulate missing sentence-transformers package."""
        from context_db.embeddings.providers.local import LocalEmbedder

        e = LocalEmbedder()
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                e._load()

    def test_load_sets_model_and_dimensions(self) -> None:
        import numpy as np
        from context_db.embeddings.providers.local import LocalEmbedder

        mock_st = _make_mock_st(dims=8)
        mock_cls = MagicMock(return_value=mock_st)

        with patch.dict(sys.modules, {"sentence_transformers": MagicMock(SentenceTransformer=mock_cls)}):
            e = LocalEmbedder("BAAI/bge-small-en-v1.5")
            e._load()

        assert e._model is mock_st
        assert e._dimensions == 8

    def test_trust_remote_code_for_nomic(self) -> None:
        from context_db.embeddings.providers.local import LocalEmbedder

        mock_st = _make_mock_st()
        mock_cls = MagicMock(return_value=mock_st)
        mock_module = MagicMock(SentenceTransformer=mock_cls)

        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            e = LocalEmbedder("nomic-ai/nomic-embed-text-v1.5")
            e._load()

        _, kwargs = mock_cls.call_args
        assert kwargs.get("trust_remote_code") is True

    def test_no_trust_remote_code_for_bge(self) -> None:
        from context_db.embeddings.providers.local import LocalEmbedder

        mock_st = _make_mock_st()
        mock_cls = MagicMock(return_value=mock_st)
        mock_module = MagicMock(SentenceTransformer=mock_cls)

        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            e = LocalEmbedder("BAAI/bge-small-en-v1.5")
            e._load()

        _, kwargs = mock_cls.call_args
        assert kwargs.get("trust_remote_code") is False

    def test_load_is_idempotent(self) -> None:
        from context_db.embeddings.providers.local import LocalEmbedder

        mock_st = _make_mock_st()
        mock_cls = MagicMock(return_value=mock_st)
        mock_module = MagicMock(SentenceTransformer=mock_cls)

        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            e = LocalEmbedder("BAAI/bge-small-en-v1.5")
            e._load()
            e._load()  # second call must not reload

        assert mock_cls.call_count == 1


# ---------------------------------------------------------------------------
# LocalEmbedder — embed / embed_batch
# ---------------------------------------------------------------------------


class TestLocalEmbedderEmbed:
    def _make_embedder(self, dims: int = 4):
        from context_db.embeddings.providers.local import LocalEmbedder

        mock_st = _make_mock_st(dims)
        mock_cls = MagicMock(return_value=mock_st)
        mock_module = MagicMock(SentenceTransformer=mock_cls)

        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            e = LocalEmbedder("BAAI/bge-small-en-v1.5")
            e._load()

        return e

    def test_embed_returns_list_of_floats(self) -> None:
        e = self._make_embedder()
        vec = e.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 4
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch_returns_correct_count(self) -> None:
        e = self._make_embedder()
        results = e.embed_batch(["a", "b", "c"])
        assert len(results) == 3

    def test_embed_batch_empty(self) -> None:
        e = self._make_embedder()
        assert e.embed_batch([]) == []

    def test_embed_batch_calls_progress_callback(self) -> None:
        from context_db.embeddings.providers.local import LocalEmbedder, _BATCH_SIZE

        mock_st = _make_mock_st(dims=2)
        mock_cls = MagicMock(return_value=mock_st)
        mock_module = MagicMock(SentenceTransformer=mock_cls)

        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            e = LocalEmbedder("BAAI/bge-small-en-v1.5")
            e._load()

        # Two full batches + one remainder.
        texts = [f"text {i}" for i in range(_BATCH_SIZE * 2 + 3)]
        calls: list[tuple[int, int]] = []
        e.embed_batch(texts, progress_callback=lambda d, t: calls.append((d, t)))
        assert len(calls) == 3
        assert calls[-1][0] == len(texts)

    def test_dimensions_property_triggers_load(self) -> None:
        from context_db.embeddings.providers.local import LocalEmbedder

        mock_st = _make_mock_st(dims=6)
        mock_cls = MagicMock(return_value=mock_st)
        mock_module = MagicMock(SentenceTransformer=mock_cls)

        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            e = LocalEmbedder("BAAI/bge-small-en-v1.5")
            assert e._model is None
            dims = e.dimensions
            assert dims == 6
            assert e._model is mock_st


# ---------------------------------------------------------------------------
# LocalEmbedder — retry logic
# ---------------------------------------------------------------------------


class TestLocalEmbedderRetry:
    def _loaded_embedder(self, dims: int = 2):
        """Return a LocalEmbedder that has already been loaded with a clean mock."""
        from context_db.embeddings.providers.local import LocalEmbedder

        mock_st = _make_mock_st(dims)
        mock_cls = MagicMock(return_value=mock_st)
        mock_module = MagicMock(SentenceTransformer=mock_cls)

        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            e = LocalEmbedder("BAAI/bge-small-en-v1.5")
            e._load()

        return e, mock_st

    def test_succeeds_after_transient_failure(self) -> None:
        import numpy as np

        e, mock_st = self._loaded_embedder()

        call_count = 0

        def flaky_encode(texts, convert_to_numpy=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return np.ones((len(texts), 2), dtype="float32")

        # Replace side_effect *after* load so the probe call is unaffected.
        mock_st.encode.side_effect = flaky_encode

        with patch("context_db.embeddings.providers.local.time.sleep"):
            results = e.embed_batch(["hello"])

        assert call_count == 2
        assert len(results) == 1

    def test_raises_after_max_retries_exhausted(self) -> None:
        from context_db.embeddings.providers.local import _MAX_RETRIES

        e, mock_st = self._loaded_embedder()

        # Reset the probe call count, then install the always-failing side_effect.
        mock_st.encode.reset_mock()
        mock_st.encode.side_effect = RuntimeError("persistent error")

        with patch("context_db.embeddings.providers.local.time.sleep"):
            with pytest.raises(RuntimeError, match="Embedding failed"):
                e.embed_batch(["hello"])

        assert mock_st.encode.call_count == _MAX_RETRIES
