"""Embedder abstract base class and provider factory.

All embedding providers implement :class:`Embedder`.  Use
:func:`create_embedder` to obtain a provider instance; this keeps call
sites independent of the concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class Embedder(ABC):
    """Provider-agnostic interface for dense text embeddings."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier stored alongside embeddings in the DB."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in every output vector."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a unit-normalised embedding vector for *text*."""
        ...

    def embed_batch(
        self,
        texts: list[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Embed a list of texts, reporting progress via *progress_callback(done, total)*.

        The default implementation calls :meth:`embed` sequentially.
        Subclasses should override this for batched inference.
        """
        results: list[list[float]] = []
        total = len(texts)
        for i, text in enumerate(texts):
            results.append(self.embed(text))
            if progress_callback:
                progress_callback(i + 1, total)
        return results


def create_embedder(model_name: str | None = None) -> Embedder:
    """Return a :class:`~context_db.embeddings.providers.local.LocalEmbedder`.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to :data:`DEFAULT_MODEL`.

    Raises
    ------
    ImportError
        When *sentence-transformers* is not installed.
    """
    from context_db.embeddings.providers.local import LocalEmbedder

    return LocalEmbedder(model_name or DEFAULT_MODEL)
