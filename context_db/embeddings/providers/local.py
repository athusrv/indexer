"""Local embedding provider backed by sentence-transformers.

Models run entirely on the local machine — no API calls, no internet
required after the initial model download.

Supported models
----------------
* ``nomic-ai/nomic-embed-text-v1.5``  (default, 768-dim, requires trust_remote_code)
* ``BAAI/bge-small-en-v1.5``          (384-dim, no trust_remote_code needed)

Any other HuggingFace sentence-transformers model can be passed as
``model_name``; the provider will auto-detect ``trust_remote_code``
based on the model namespace.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog

from context_db.embeddings.embedder import Embedder

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BATCH_SIZE = 32

# Models that require trust_remote_code=True when loaded via sentence-transformers.
_TRUST_REMOTE_PREFIXES = ("nomic-ai/",)


class LocalEmbedder(Embedder):
    """Runs a sentence-transformers model locally on CPU or GPU.

    The underlying model is loaded lazily on first use, so constructing
    a :class:`LocalEmbedder` is always cheap.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to ``nomic-ai/nomic-embed-text-v1.5``.
    """

    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    ALT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._dimensions: int | None = None

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._load()
        assert self._dimensions is not None
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: list[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Embed *texts* in sub-batches of :data:`_BATCH_SIZE`, with retries.

        Parameters
        ----------
        texts:
            Strings to embed.  Empty list returns ``[]`` immediately.
        progress_callback:
            Optional ``callback(done: int, total: int)`` called after each
            sub-batch completes.
        """
        if not texts:
            return []

        self._load()

        results: list[list[float]] = []
        total = len(texts)

        for start in range(0, total, _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            vecs = self._encode_with_retry(batch)
            results.extend(vecs)
            if progress_callback:
                progress_callback(min(start + _BATCH_SIZE, total), total)

        logger.debug("embedded_batch", model=self._model_name, count=total)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazily load the sentence-transformers model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install 'context-db[embeddings]'"
            ) from exc

        trust_remote = any(
            self._model_name.startswith(prefix) for prefix in _TRUST_REMOTE_PREFIXES
        )

        logger.info("loading_embedding_model", model=self._model_name)
        self._model = SentenceTransformer(
            self._model_name,
            trust_remote_code=trust_remote,
        )

        # Probe dimensions with a single dummy encode.
        probe = self._model.encode(["probe"], convert_to_numpy=True)
        self._dimensions = int(probe.shape[1])

        logger.info(
            "embedding_model_loaded",
            model=self._model_name,
            dimensions=self._dimensions,
        )

    def _encode_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Run ``model.encode`` with exponential-backoff retry."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                vecs = self._model.encode(texts, convert_to_numpy=True)  # type: ignore[union-attr]
                return [v.tolist() for v in vecs]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    sleep_s = 2**attempt
                    logger.warning(
                        "embed_retry",
                        attempt=attempt + 1,
                        sleep_s=sleep_s,
                        error=str(exc),
                    )
                    time.sleep(sleep_s)
        raise RuntimeError(
            f"Embedding failed after {_MAX_RETRIES} attempts"
        ) from last_exc
