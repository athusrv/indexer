"""Semantic retrieval via dense-vector cosine similarity.

Search flow
-----------
query → embed → load all stored vectors → cosine similarity → top-k

This is a brute-force O(N·D) implementation — acceptable for codebases up to
~100 k chunks.  No ANN index or external vector store is required.

When *numpy* is available (it always is when sentence-transformers is
installed) the hot path is vectorised with a matrix multiply.  A pure-Python
fallback is provided so the module can be imported without numpy.

Result format
-------------
Returns a list of :class:`SemanticHit` objects:

    [SemanticHit(chunk_id=42, score=0.87), ...]

Scores are cosine similarities in [-1, 1]; in practice they are in [0, 1] for
models that produce non-negative embeddings.  Results are sorted descending.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import structlog

from context_db.embeddings.embedder import Embedder
from context_db.storage.repository import Repository

logger = structlog.get_logger(__name__)

_PREVIEW_CHARS: int = 300


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticHit:
    """A single semantic search result — chunk id plus cosine score."""

    chunk_id: int
    score: float  # cosine similarity, higher is better


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def semantic_search(
    query: str,
    *,
    embedder: Embedder,
    repo: Repository,
    limit: int = 20,
) -> list[SemanticHit]:
    """Rank all indexed chunks by cosine similarity to *query*.

    Parameters
    ----------
    query:
        Natural-language query string.
    embedder:
        An :class:`~context_db.embeddings.embedder.Embedder` instance; its
        :attr:`model_name` selects which stored vectors to compare against.
    repo:
        Open :class:`~context_db.storage.repository.Repository` instance.
    limit:
        Maximum number of results to return.

    Returns
    -------
    list[SemanticHit]
        At most *limit* hits, sorted by score descending.  Empty when no
        embeddings exist or the query is blank.
    """
    if not query.strip():
        return []

    # 1. Embed the query.
    query_vec = embedder.embed(query)

    # 2. Load all stored embeddings for this model.
    pairs = repo.get_all_embeddings(embedder.model_name)

    if not pairs:
        logger.debug("semantic_search_no_embeddings", model=embedder.model_name)
        return []

    # 3. Cosine similarity + top-k selection.
    hits = _cosine_top_k(query_vec, pairs, limit=limit)

    logger.debug(
        "semantic_search_done",
        model=embedder.model_name,
        candidates=len(pairs),
        results=len(hits),
    )
    return hits


# ---------------------------------------------------------------------------
# Cosine similarity — numpy fast path + pure-Python fallback
# ---------------------------------------------------------------------------


def _cosine_top_k(
    query_vec: list[float],
    pairs: list[tuple[int, list[float]]],
    limit: int,
) -> list[SemanticHit]:
    """Dispatch to numpy or pure-Python implementation."""
    try:
        import numpy as np  # noqa: F401 — presence check only
        return _cosine_top_k_numpy(query_vec, pairs, limit)
    except ImportError:
        return _cosine_top_k_pure(query_vec, pairs, limit)


def _cosine_top_k_numpy(
    query_vec: list[float],
    pairs: list[tuple[int, list[float]]],
    limit: int,
) -> list[SemanticHit]:
    """Vectorised cosine top-k using numpy matrix multiply."""
    import numpy as np

    chunk_ids = [p[0] for p in pairs]
    matrix = np.array([p[1] for p in pairs], dtype="float32")  # (N, D)
    q = np.array(query_vec, dtype="float32")  # (D,)

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    q = q / q_norm

    row_norms = np.linalg.norm(matrix, axis=1)  # (N,)
    nonzero = row_norms != 0.0
    matrix[nonzero] = matrix[nonzero] / row_norms[nonzero, np.newaxis]

    scores = matrix @ q  # (N,)

    k = min(limit, len(scores))
    # argpartition is O(N); argsort of the k winners is O(k log k).
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    return [
        SemanticHit(chunk_id=chunk_ids[int(i)], score=float(scores[i]))
        for i in top_idx
    ]


def _cosine_top_k_pure(
    query_vec: list[float],
    pairs: list[tuple[int, list[float]]],
    limit: int,
) -> list[SemanticHit]:
    """Pure-Python fallback — no external dependencies."""
    q_norm = math.sqrt(sum(x * x for x in query_vec))
    if q_norm == 0.0:
        return []

    hits: list[SemanticHit] = []
    for chunk_id, vec in pairs:
        v_norm = math.sqrt(sum(x * x for x in vec))
        if v_norm == 0.0:
            continue
        dot = sum(a * b for a, b in zip(query_vec, vec))
        hits.append(SemanticHit(chunk_id=chunk_id, score=dot / (q_norm * v_norm)))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]
