"""Hybrid retrieval — weighted combination of lexical (FTS5) and semantic search.

Search flow
-----------
1.  FTS5 chunk-level search  → top-50 candidates  (BM25 scores)
2.  Dense vector search      → top-50 candidates  (cosine scores)
3.  Union the two candidate sets (deduplicated on chunk_id)
4.  Normalise each score set independently to [0, 1] (min-max)
5.  Combine:  final = 0.60 × lex_norm + 0.40 × sem_norm
    Chunks that appear in only one set receive 0.0 for the missing signal.
6.  Sort descending, return top-*limit* as :class:`~context_db.models.HybridResult`

Match type
----------
* ``"lexical"``  — found only by FTS5
* ``"semantic"`` — found only by embedding search
* ``"hybrid"``   — found by both; score reflects both signals

Scoring weights (module-level constants, easy to tune)
-------------------------------------------------------
``_LEXICAL_WEIGHT = 0.60``
``_SEMANTIC_WEIGHT = 0.40``
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import structlog

from context_db.embeddings.embedder import Embedder
from context_db.models import HybridResult
from context_db.retrieval.semantic import semantic_search
from context_db.storage.repository import Repository

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

_LEXICAL_CANDIDATES: int = 50
_SEMANTIC_CANDIDATES: int = 50
_LEXICAL_WEIGHT: float = 0.60
_SEMANTIC_WEIGHT: float = 0.40
_PREVIEW_CHARS: int = 300

# FTS5 operators — queries containing these are forwarded verbatim.
_FTS_OPERATORS: frozenset[str] = frozenset({"AND", "OR", "NOT"})
_FTS_SPECIAL_CHARS: str = '"*():-'


# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LexicalHit:
    chunk_id: int
    path: Path
    start_line: int
    end_line: int
    content: str
    score: float


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    *,
    embedder: Embedder,
    repo: Repository,
    conn: sqlite3.Connection,
    limit: int = 20,
    preview_chars: int = _PREVIEW_CHARS,
) -> list[HybridResult]:
    """Return *limit* results ranked by the weighted combination of BM25 + cosine.

    Parameters
    ----------
    query:
        Natural-language query string.
    embedder:
        Embedder instance; its :attr:`model_name` selects stored vectors.
    repo:
        Open :class:`~context_db.storage.repository.Repository`.
    conn:
        The underlying SQLite connection (for the FTS query).
    limit:
        Maximum number of results.
    preview_chars:
        Characters of chunk content to include in each result's ``preview``.

    Returns
    -------
    list[HybridResult]
        Results sorted by final score descending.  Empty when the query is
        blank or no candidates are found by either retrieval path.
    """
    if not query.strip():
        return []

    # ── 1. Lexical candidates ─────────────────────────────────────────────
    lex_hits: dict[int, _LexicalHit] = {
        h.chunk_id: h
        for h in _fts_chunk_search(query, conn, limit=_LEXICAL_CANDIDATES)
    }
    lex_scores: dict[int, float] = {cid: h.score for cid, h in lex_hits.items()}

    # ── 2. Semantic candidates ────────────────────────────────────────────
    sem_raw = semantic_search(query, embedder=embedder, repo=repo, limit=_SEMANTIC_CANDIDATES)
    sem_scores: dict[int, float] = {h.chunk_id: h.score for h in sem_raw}

    # ── 3. Union candidate set ────────────────────────────────────────────
    all_ids = set(lex_scores) | set(sem_scores)
    if not all_ids:
        return []

    # ── 4. Normalise independently ────────────────────────────────────────
    lex_norm = _normalize(lex_scores)
    sem_norm = _normalize(sem_scores)

    # ── 5. Score + determine match type ──────────────────────────────────
    scored: list[tuple[int, float, str]] = []
    for chunk_id in all_ids:
        nl = lex_norm.get(chunk_id, 0.0)
        ns = sem_norm.get(chunk_id, 0.0)
        final = _LEXICAL_WEIGHT * nl + _SEMANTIC_WEIGHT * ns

        has_lex = chunk_id in lex_scores
        has_sem = chunk_id in sem_scores
        if has_lex and has_sem:
            match_type: str = "hybrid"
        elif has_lex:
            match_type = "lexical"
        else:
            match_type = "semantic"

        scored.append((chunk_id, final, match_type))

    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:limit]

    # ── 6. Hydrate chunk info ─────────────────────────────────────────────
    # Use already-fetched FTS data for lexical hits; query DB for the rest.
    chunk_info: dict[int, _LexicalHit] = {**lex_hits}
    missing = [cid for cid, _, _ in top if cid not in chunk_info]
    if missing:
        for info in repo.get_chunk_file_info(missing):
            chunk_info[info.chunk_id] = _LexicalHit(
                chunk_id=info.chunk_id,
                path=info.path,
                start_line=info.start_line,
                end_line=info.end_line,
                content=info.content,
                score=0.0,  # placeholder — not used after this point
            )

    results: list[HybridResult] = []
    for chunk_id, final_score, match_type in top:
        info = chunk_info.get(chunk_id)
        if info is None:
            continue  # chunk was deleted between query and hydration
        results.append(
            HybridResult(
                path=info.path,
                score=round(final_score, 6),
                match_type=match_type,  # type: ignore[arg-type]
                start_line=info.start_line,
                end_line=info.end_line,
                preview=info.content[:preview_chars],
            )
        )

    logger.debug(
        "hybrid_search_done",
        query=query,
        lexical_candidates=len(lex_scores),
        semantic_candidates=len(sem_scores),
        results=len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_fts_query(query: str) -> str:
    """Prepare a raw query string for FTS5's MATCH operator.

    Mirrors :meth:`~context_db.retrieval.search.SearchEngine._prepare_query`
    so that hybrid search and lexical search handle queries identically.
    """
    words = query.split()
    if any(w in _FTS_OPERATORS for w in words) or any(ch in query for ch in _FTS_SPECIAL_CHARS):
        return query
    if len(words) == 1:
        return query
    escaped = query.replace('"', '""')
    return f'"{escaped}"'


def _fts_chunk_search(
    query: str,
    conn: sqlite3.Connection,
    limit: int,
) -> list[_LexicalHit]:
    """Run a chunk-level BM25 search, returning chunk_id alongside results."""
    fts_query = _prepare_fts_query(query)
    try:
        rows = conn.execute(
            """
            SELECT
                c.id              AS chunk_id,
                f.path,
                c.start_line,
                c.end_line,
                c.content,
                -bm25(chunks_fts) AS score
            FROM  chunks_fts
            JOIN  chunks c ON chunks_fts.rowid = c.id
            JOIN  files  f ON c.file_id = f.id
            WHERE chunks_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("fts_chunk_search_error", query=query, error=str(exc))
        return []

    return [
        _LexicalHit(
            chunk_id=row["chunk_id"],
            path=Path(row["path"]),
            start_line=row["start_line"],
            end_line=row["end_line"],
            content=row["content"],
            score=float(row["score"]),
        )
        for row in rows
    ]


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalise *scores* to [0, 1].

    When all values are identical the normalised score is 1.0 for every
    entry — they all tied at maximum relevance within their set.
    """
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}
