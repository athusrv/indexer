"""Full-text search engine backed by SQLite FTS5 + BM25 ranking.

SQLite's FTS5 module exposes a ``bm25()`` function that returns a *negative*
score (lower is better) — we negate it so that higher scores are better,
matching the conventional expectation of retrieval APIs.

Deduplication
-------------
Files are split into multiple chunks by the indexer.  A single query can
match many chunks from the same file, which would cause the same file to
appear repeatedly in results.  By default, ``search()`` deduplicates results
so each file appears **at most once**, represented by its highest-scoring
chunk.  Pass ``deduplicate=False`` to get all chunk-level hits instead (useful
when building multi-chunk context windows for LLMs).

Query language
--------------
Queries are forwarded verbatim to FTS5's MATCH operator, which supports:
* Simple phrase matching: ``"jwt auth"``
* Prefix search:         ``auth*``
* Column filters:        ``content:token``
* Boolean operators:     ``AND``, ``OR``, ``NOT`` (upper-case)

Callers do not need to quote queries — single-word queries work as-is.
Multi-word queries are wrapped in quotes automatically unless the caller has
already used FTS5 operators.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

from context_db.models import SearchResult

logger = structlog.get_logger(__name__)

_PREVIEW_CHARS: int = 300   # characters shown in the preview field
_DEFAULT_LIMIT: int = 20

# How many raw chunk rows to fetch before deduplication.  A factor of 10
# ensures we almost always have enough unique files even for large codebases.
_OVER_FETCH_FACTOR: int = 10


class SearchEngine:
    """BM25-ranked full-text search over indexed chunks.

    Parameters
    ----------
    conn:
        An open SQLite connection (from :func:`context_db.storage.db.open_db`).
    preview_chars:
        Number of characters to return in the ``preview`` field of each result.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        preview_chars: int = _PREVIEW_CHARS,
    ) -> None:
        self._conn = conn
        self._preview_chars = preview_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = _DEFAULT_LIMIT,
        path_filter: str | None = None,
        deduplicate: bool = True,
    ) -> list[SearchResult]:
        """Search the FTS index and return ranked results.

        Parameters
        ----------
        query:
            Search terms.  Plain words, quoted phrases, or FTS5 expressions.
        limit:
            Maximum number of results to return.  When *deduplicate* is
            ``True`` (the default) this is the number of **unique files**
            returned.
        path_filter:
            Optional SQL ``LIKE`` pattern applied to the file path
            (e.g. ``"%.py"`` to restrict to Python files).
        deduplicate:
            When ``True`` (default), each file appears at most once in the
            results — represented by its highest-scoring chunk.
            Set to ``False`` to receive all matching chunk-level hits.

        Returns
        -------
        list[SearchResult]
            Results ordered by BM25 relevance (best first).
        """
        if not query.strip():
            return []

        fts_query = self._prepare_query(query)
        logger.debug(
            "search",
            query=query,
            fts_query=fts_query,
            limit=limit,
            deduplicate=deduplicate,
        )

        try:
            # Over-fetch so deduplication still yields `limit` unique files.
            fetch_limit = limit * _OVER_FETCH_FACTOR if deduplicate else limit
            raw = self._execute_search(
                fts_query, limit=fetch_limit, path_filter=path_filter
            )
            if deduplicate:
                return self._deduplicate(raw, limit=limit)
            return raw
        except sqlite3.OperationalError as exc:
            # FTS5 syntax errors surface here — return empty rather than crash.
            logger.warning("search_error", query=query, error=str(exc))
            return []

    def suggest_terms(self, prefix: str, *, limit: int = 10) -> list[str]:
        """Return FTS5 vocabulary terms starting with *prefix*.

        Useful for auto-complete features.
        """
        if not prefix.strip():
            return []
        rows = self._conn.execute(
            """
            SELECT term FROM chunks_fts_vocab
            WHERE term LIKE ?
            ORDER BY doc DESC
            LIMIT ?
            """,
            (f"{prefix}%", limit),
        ).fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(results: list[SearchResult], *, limit: int) -> list[SearchResult]:
        """Keep only the highest-scoring chunk per file path.

        ``results`` must already be sorted by score descending (as returned by
        ``_execute_search``).  Iteration order therefore guarantees that the
        first time we see a path it carries the best score for that file.
        """
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for r in results:
            key = str(r.path)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
                if len(deduped) == limit:
                    break
        return deduped

    def _prepare_query(self, query: str) -> str:
        """Wrap plain multi-word queries in double quotes for phrase search.

        If the query contains FTS5 operators (AND, OR, NOT, *, ") it is
        passed through unchanged so the caller retains full control.
        """
        fts_operators = {"AND", "OR", "NOT"}
        words = query.split()
        has_operator = any(w in fts_operators for w in words)
        has_special = any(ch in query for ch in '"*():-')

        if has_operator or has_special:
            return query

        # For a single word, just return it as-is (no quoting needed).
        if len(words) == 1:
            return query

        # Wrap multi-word plain queries so FTS5 treats them as a phrase.
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def _execute_search(
        self,
        fts_query: str,
        *,
        limit: int,
        path_filter: str | None,
    ) -> list[SearchResult]:
        """Execute the FTS5 query and hydrate :class:`SearchResult` objects."""
        if path_filter:
            sql = """
                SELECT
                    f.path,
                    -bm25(chunks_fts)   AS score,
                    c.start_line,
                    c.end_line,
                    c.content
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                JOIN files  f ON c.file_id = f.id
                WHERE chunks_fts MATCH ?
                  AND f.path LIKE ?
                ORDER BY score DESC
                LIMIT ?
            """
            params: tuple = (fts_query, path_filter, limit)
        else:
            sql = """
                SELECT
                    f.path,
                    -bm25(chunks_fts)   AS score,
                    c.start_line,
                    c.end_line,
                    c.content
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                JOIN files  f ON c.file_id = f.id
                WHERE chunks_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
            """
            params = (fts_query, limit)

        rows = self._conn.execute(sql, params).fetchall()

        results: list[SearchResult] = []
        for row in rows:
            preview = row["content"][: self._preview_chars]
            results.append(
                SearchResult(
                    path=Path(row["path"]),
                    score=float(row["score"]),
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    preview=preview,
                )
            )

        logger.debug("search_results", count=len(results))
        return results
