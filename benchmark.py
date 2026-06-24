#!/usr/bin/env python3
"""Retrieval benchmark: FTS-only vs Semantic-only vs Hybrid.

Usage
-----
    uv run python benchmark.py [DIRECTORY]

DIRECTORY defaults to the current working directory.  The script creates a
temporary database, indexes the directory, optionally generates embeddings
(when sentence-transformers is installed), then runs each retrieval strategy
against a set of sample queries and reports latency, index size, and example
results.

Output example
--------------
    ╭─ Benchmark Results ─────────────────────────────────────────────╮
    │ Directory      /Users/me/project                                 │
    │ Files          42   Chunks  318                                  │
    │ DB size        1.2 MB                                            │
    ╰─────────────────────────────────────────────────────────────────╯

    Strategy        Avg latency    P95 latency    Results/query
    ──────────────────────────────────────────────────────────
    FTS only        1.3 ms         2.1 ms         4.2
    Semantic only   8.4 ms         11.2 ms        5.0
    Hybrid          9.7 ms         13.0 ms        6.1
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, quantiles
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()

SAMPLE_QUERIES = [
    "authentication",
    "error handling",
    "database connection",
    "test fixture",
    "import",
    "configuration",
    "logging",
    "type annotation",
]

_SEARCH_LIMIT = 10


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Human-readable helpers
# ---------------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_fts(conn: Any, queries: list[str]) -> tuple[list[float], dict[str, list]]:
    from context_db.retrieval.search import SearchEngine

    engine = SearchEngine(conn)
    latencies: list[float] = []
    results: dict[str, list] = {}
    for q in queries:
        hits, lat = _timed(engine.search, q, limit=_SEARCH_LIMIT, deduplicate=False)
        latencies.append(lat)
        results[q] = hits
    return latencies, results


def _run_semantic(embedder: Any, repo: Any, queries: list[str]) -> tuple[list[float], dict[str, list]]:
    from context_db.retrieval.semantic import semantic_search

    latencies: list[float] = []
    results: dict[str, list] = {}
    for q in queries:
        hits, lat = _timed(semantic_search, q, embedder=embedder, repo=repo, limit=_SEARCH_LIMIT)
        latencies.append(lat)
        results[q] = hits
    return latencies, results


def _run_hybrid(embedder: Any, repo: Any, conn: Any, queries: list[str]) -> tuple[list[float], dict[str, list]]:
    from context_db.retrieval.hybrid import hybrid_search

    latencies: list[float] = []
    results: dict[str, list] = {}
    for q in queries:
        hits, lat = _timed(hybrid_search, q, embedder=embedder, repo=repo, conn=conn, limit=_SEARCH_LIMIT)
        latencies.append(lat)
        results[q] = hits
    return latencies, results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _print_header(root: Path, file_count: int, chunk_count: int, db_size: int) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Directory", str(root))
    table.add_row("Files indexed", str(file_count))
    table.add_row("Chunks", str(chunk_count))
    table.add_row("DB size", _human_bytes(db_size))
    console.print(Panel(table, title="[bold]Benchmark Setup[/bold]", expand=False))
    console.print()


def _print_latency_table(strategies: list[tuple[str, list[float], dict[str, list]]]) -> None:
    console.print(Rule("[bold]Latency Summary[/bold]"))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Strategy", style="cyan")
    table.add_column("Avg latency", justify="right")
    table.add_column("P95 latency", justify="right")
    table.add_column("Avg results/query", justify="right")

    for name, latencies, results in strategies:
        avg = mean(latencies) if latencies else 0.0
        p95 = quantiles(latencies, n=20)[18] if len(latencies) >= 2 else latencies[0] if latencies else 0.0
        avg_hits = mean(len(v) for v in results.values()) if results else 0.0
        table.add_row(name, _ms(avg), _ms(p95), f"{avg_hits:.1f}")

    console.print(table)
    console.print()


def _print_examples(
    query: str,
    fts_hits: list,
    sem_hits: list | None,
    hybrid_hits: list | None,
) -> None:
    console.print(Rule(f'[bold]Example query:[/bold] [yellow]"{query}"[/yellow]'))

    def _fmt_fts(hits: list) -> None:
        for h in hits[:3]:
            console.print(f"  [cyan]{h.path.name}[/cyan] L{h.start_line} score={h.score:.4f}")
            console.print(f"    [dim]{h.preview[:80].strip()}[/dim]")

    def _fmt_sem(hits: list) -> None:
        for h in hits[:3]:
            console.print(f"  chunk_id={h.chunk_id} score={h.score:.4f}")

    def _fmt_hybrid(hits: list) -> None:
        for h in hits[:3]:
            console.print(f"  [cyan]{h.path.name}[/cyan] [{h.match_type}] score={h.score:.4f}")
            console.print(f"    [dim]{h.preview[:80].strip()}[/dim]")

    console.print("[bold]  FTS only:[/bold]")
    _fmt_fts(fts_hits)
    if sem_hits is not None:
        console.print("[bold]  Semantic:[/bold]")
        _fmt_sem(sem_hits)
    if hybrid_hits is not None:
        console.print("[bold]  Hybrid:[/bold]")
        _fmt_hybrid(hybrid_hits)
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(directory: str | None = None) -> None:
    root = Path(directory or ".").resolve()
    if not root.exists():
        console.print(f"[red]Directory not found:[/red] {root}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "bench.db"

        # ── Index ─────────────────────────────────────────────────────────
        console.print(f"[bold]Indexing[/bold] {root} ...", end=" ")
        sys.stdout.flush()

        from context_db.indexer.chunker import Chunker
        from context_db.indexer.pipeline import IndexingPipeline
        from context_db.indexer.scanner import Scanner
        from context_db.storage.db import open_db
        from context_db.storage.repository import Repository

        conn = open_db(db_path)
        repo = Repository(conn)
        pipeline = IndexingPipeline(repository=repo, scanner=Scanner(), chunker=Chunker())
        index_result, index_lat = _timed(pipeline.run, root)
        console.print(f"done in {index_lat:.1f}s ({index_result.total_chunks} chunks)")

        stats = repo.get_stats(db_path)
        _print_header(root, stats.file_count, stats.chunk_count, stats.db_size_bytes)

        # ── FTS benchmark ─────────────────────────────────────────────────
        fts_latencies, fts_results = _run_fts(conn, SAMPLE_QUERIES)

        # ── Semantic + Hybrid benchmark (optional) ────────────────────────
        sem_latencies: list[float] = []
        hybrid_latencies: list[float] = []
        sem_results: dict[str, list] = {}
        hybrid_results: dict[str, list] = {}
        embedder = None

        try:
            from context_db.embeddings import create_embedder

            console.print("[bold]Loading embedding model[/bold] ...", end=" ")
            embedder = create_embedder()
            console.print(f"[green]{embedder.model_name}[/green] ({embedder.dimensions}d)")

            console.print("[bold]Generating embeddings[/bold] ...", end=" ")
            missing = repo.get_chunks_without_embeddings(embedder.model_name)
            if missing:
                vectors = embedder.embed_batch([c.content for c in missing])
                repo.upsert_embeddings_batch(
                    [c.id for c in missing], vectors, embedder.model_name
                )
            emb_count = repo.get_embedding_count(embedder.model_name)
            console.print(f"done ({emb_count} vectors)")
            console.print()

            sem_latencies, sem_results = _run_semantic(embedder, repo, SAMPLE_QUERIES)
            hybrid_latencies, hybrid_results = _run_hybrid(embedder, repo, conn, SAMPLE_QUERIES)

        except ImportError:
            console.print(
                "\n[yellow]sentence-transformers not installed — "
                "semantic/hybrid benchmarks skipped.[/yellow]\n"
                "Install with:  pip install 'context-db[embeddings]'\n"
            )

        # ── Report ────────────────────────────────────────────────────────
        strategies: list[tuple[str, list[float], dict[str, list]]] = [
            ("FTS only", fts_latencies, fts_results),
        ]
        if embedder is not None:
            strategies += [
                ("Semantic only", sem_latencies, sem_results),
                ("Hybrid", hybrid_latencies, hybrid_results),
            ]

        _print_latency_table(strategies)

        # Print examples for the first query that returned FTS results.
        example_query = next(
            (q for q in SAMPLE_QUERIES if fts_results.get(q)),
            SAMPLE_QUERIES[0],
        )
        _print_examples(
            example_query,
            fts_results.get(example_query, []),
            sem_results.get(example_query) if embedder else None,
            hybrid_results.get(example_query) if embedder else None,
        )

        conn.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
