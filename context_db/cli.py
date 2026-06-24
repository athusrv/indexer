"""CLI entry-point for context-db.

Commands
--------
ctx init                – Initialise a new database in the current directory.
ctx index <PATH>        – Index (or re-index) a directory tree.
ctx search <QUERY>      – Search the index with BM25, semantic, or hybrid ranking.
ctx stats               – Print index statistics.
ctx reset               – Wipe the index and start fresh.

The database file defaults to ``./context.db`` but can be overridden via the
``--db`` global option or the ``CTX_DB`` environment variable.

Environment variables
---------------------
CTX_DB            Path to the SQLite database  (default: ./context.db)
CTX_LOG_LEVEL     Numeric log level             (default: 30 = WARNING)
CTX_EMBED_MODEL   Default embedding model ID    (default: nomic-ai/nomic-embed-text-v1.5)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from context_db.embeddings.embedder import DEFAULT_MODEL, create_embedder
from context_db.indexer.chunker import Chunker
from context_db.indexer.pipeline import IndexingPipeline, ProgressEvent
from context_db.indexer.scanner import Scanner
from context_db.retrieval.hybrid import hybrid_search
from context_db.retrieval.search import SearchEngine
from context_db.retrieval.semantic import semantic_search
from context_db.storage.db import open_db
from context_db.storage.repository import Repository

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(level: int) -> None:
    """Configure structlog at the given numeric log level.

    All log output is directed to **stderr** so that stdout stays clean for
    machine-readable output (``--json``, pipes to ``jq``, etc.).
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


# Configure once at import time using the env var (WARNING by default).
_configure_logging(int(os.environ.get("CTX_LOG_LEVEL", "30")))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ctx",
    help="context-db — filesystem indexing engine for AI agents.",
    add_completion=False,
    rich_markup_mode="rich",
)
# stdout  — formatted / data output (pretty results, stats tables)
# stderr  — status messages, warnings, errors, progress
console = Console()
err_console = Console(stderr=True)

_DEFAULT_DB = Path("context.db")


def _db_path(db: Optional[Path]) -> Path:
    """Resolve DB path: CLI flag → env var → default."""
    if db:
        return db
    env = os.environ.get("CTX_DB")
    if env:
        return Path(env)
    return _DEFAULT_DB


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    db: Annotated[Optional[Path], typer.Option("--db", help="Database file path.")] = None,
) -> None:
    """Initialise a new context-db database."""
    path = _db_path(db)
    if path.exists():
        console.print(f"[yellow]Database already exists:[/yellow] {path}")
        raise typer.Exit(0)

    conn = open_db(path)
    conn.close()
    console.print(f"[green]Initialised database:[/green] {path}")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@app.command()
def index(
    directory: Annotated[Path, typer.Argument(help="Directory to index.")] = Path("."),
    db: Annotated[Optional[Path], typer.Option("--db", help="Database file path.")] = None,
    chunk_chars: Annotated[int, typer.Option("--chunk-chars", help="Target chars per chunk.")] = 1_500,
    overlap: Annotated[int, typer.Option("--overlap", help="Overlap lines between chunks.")] = 3,
    ignore: Annotated[Optional[list[str]], typer.Option("--ignore", help="Extra ignore patterns.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    embed_model: Annotated[Optional[str], typer.Option("--embed-model", help="Generate embeddings with this model (e.g. nomic-ai/nomic-embed-text-v1.5).")] = None,
    disable_embeddings: Annotated[bool, typer.Option("--disable-embeddings", help="Skip the embedding step even if --embed-model is set.")] = False,
    reembed: Annotated[bool, typer.Option("--reembed", help="Delete existing embeddings and regenerate from scratch.")] = False,
) -> None:
    """Index a directory tree into the database.

    Optionally generate dense vector embeddings with ``--embed-model``:

        ctx index . --embed-model nomic-ai/nomic-embed-text-v1.5

    Use ``--reembed`` to force regeneration of all embeddings.
    """
    if verbose:
        # Re-configure at DEBUG level for this process invocation.
        # In tests, use CTX_LOG_LEVEL=10 env var to avoid global mutation.
        _configure_logging(10)

    # Resolve embedder — optional, opt-in only.
    embedder = None
    if embed_model and not disable_embeddings:
        try:
            embedder = create_embedder(embed_model)
        except ImportError as exc:
            err_console.print(f"[red]Cannot load embedder:[/red] {exc}")
            raise typer.Exit(1)

    db_path = _db_path(db)
    conn = open_db(db_path)
    repo = Repository(conn)
    scanner = Scanner(ignore_patterns=ignore or [])
    chunker = Chunker(chunk_chars=chunk_chars, overlap_lines=overlap)

    # Progress bar setup
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("• {task.fields[action]}"),
        console=console,
        transient=True,
    )
    task_id = None

    def on_progress(evt: ProgressEvent) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task(
                "Indexing",
                total=evt.total,
                action="",
            )
        progress.update(
            task_id,
            completed=evt.current,
            action=f"[cyan]{evt.path.name}[/cyan]",
        )

    pipeline = IndexingPipeline(
        repository=repo,
        scanner=scanner,
        chunker=chunker,
        progress_callback=on_progress,
        embedder=embedder,
        reembed=reembed,
    )

    t0 = time.perf_counter()
    with progress:
        result = pipeline.run(directory.resolve())
    duration = time.perf_counter() - t0

    conn.close()

    # Summary panel
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Files indexed:", str(result.indexed))
    table.add_row("Files deleted:", str(result.deleted))
    table.add_row("Files skipped:", str(result.skipped))
    table.add_row("Chunks created:", str(result.total_chunks))
    if result.embedded:
        table.add_row("Chunks embedded:", str(result.embedded))
    table.add_row("Errors:", str(result.errors))
    table.add_row("Duration:", f"{duration:.1f}s")

    console.print(Panel(table, title="[bold green]Index complete[/bold green]", expand=False))

    if result.errors:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    db: Annotated[Optional[Path], typer.Option("--db", help="Database file path.")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results.")] = 10,
    path_filter: Annotated[Optional[str], typer.Option("--path", help="SQL LIKE filter on file path.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
    all_chunks: Annotated[bool, typer.Option("--all-chunks", help="Show all matching chunks, not just the best per file.")] = False,
    semantic: Annotated[bool, typer.Option("--semantic", help="Use semantic (dense vector) search only.")] = False,
    hybrid: Annotated[bool, typer.Option("--hybrid", help="Use hybrid search (FTS + semantic, weighted rerank).")] = False,
    embed_model: Annotated[Optional[str], typer.Option("--embed-model", help="Embedding model for semantic/hybrid search.")] = None,
    disable_embeddings: Annotated[bool, typer.Option("--disable-embeddings", help="Force FTS-only search, ignoring --semantic/--hybrid.")] = False,
) -> None:
    """Search the index.

    Default: BM25 lexical search (fast, no model required).

        ctx search "jwt authentication"

    Semantic search (requires embeddings to have been generated during indexing):

        ctx search "token validation" --semantic

    Hybrid search — best of both worlds:

        ctx search "error handling" --hybrid
    """
    db_path = _db_path(db)
    if not db_path.exists():
        err_console.print("[red]Database not found.[/red] Run [bold]ctx init[/bold] first.")
        raise typer.Exit(1)

    conn = open_db(db_path)
    repo = Repository(conn)

    # Determine search mode (--disable-embeddings wins if set).
    use_semantic = semantic and not disable_embeddings
    use_hybrid = hybrid and not disable_embeddings

    # Load embedder when a vector-based mode is requested.
    embedder = None
    if (use_semantic or use_hybrid):
        model = embed_model or os.environ.get("CTX_EMBED_MODEL") or DEFAULT_MODEL
        try:
            embedder = create_embedder(model)
        except ImportError as exc:
            err_console.print(f"[red]Cannot load embedder:[/red] {exc}")
            conn.close()
            raise typer.Exit(1)

    # ── Run the appropriate retrieval strategy ────────────────────────────
    if use_hybrid and embedder is not None:
        raw = hybrid_search(query, embedder=embedder, repo=repo, conn=conn, limit=limit)
        results = [
            {
                "path": str(r.path),
                "score": r.score,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "preview": r.preview,
                "match_type": r.match_type,
            }
            for r in raw
        ]
    elif use_semantic and embedder is not None:
        from context_db.models import ChunkFileInfo
        sem_hits = semantic_search(query, embedder=embedder, repo=repo, limit=limit)
        chunk_ids = [h.chunk_id for h in sem_hits]
        score_map = {h.chunk_id: h.score for h in sem_hits}
        infos: dict[int, ChunkFileInfo] = {
            i.chunk_id: i for i in repo.get_chunk_file_info(chunk_ids)
        }
        results = []
        for hit in sem_hits:
            info = infos.get(hit.chunk_id)
            if info is None:
                continue
            results.append(
                {
                    "path": str(info.path),
                    "score": hit.score,
                    "start_line": info.start_line,
                    "end_line": info.end_line,
                    "preview": info.content[:300],
                    "match_type": "semantic",
                }
            )
    else:
        # Default: FTS-only (unchanged behaviour).
        engine = SearchEngine(conn)
        fts_results = engine.search(
            query,
            limit=limit,
            path_filter=path_filter,
            deduplicate=not all_chunks,
        )
        results = [
            {
                "path": str(r.path),
                "score": r.score,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "preview": r.preview,
                "match_type": "lexical",
            }
            for r in fts_results
        ]

    conn.close()

    if not results:
        err_console.print(f"[yellow]No results for:[/yellow] {query!r}")
        raise typer.Exit(0)

    if json_output:
        import json
        typer.echo(json.dumps(results, indent=2))
        return

    for i, r in enumerate(results, 1):
        match_tag = f"[dim][{r['match_type']}][/dim] " if (use_semantic or use_hybrid) else ""
        console.rule(
            f"[bold]{i}.[/bold] {match_tag}[cyan]{r['path']}[/cyan] "
            f"[dim]lines {r['start_line']}–{r['end_line']}[/dim] "
            f"[yellow]score={r['score']:.4f}[/yellow]"
        )
        console.print(r["preview"].strip())
        console.print()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    db: Annotated[Optional[Path], typer.Option("--db", help="Database file path.")] = None,
) -> None:
    """Print statistics about the current index."""
    db_path = _db_path(db)
    if not db_path.exists():
        err_console.print("[red]Database not found.[/red] Run [bold]ctx init[/bold] first.")
        raise typer.Exit(1)

    conn = open_db(db_path)
    repo = Repository(conn)
    s = repo.get_stats(db_path)
    conn.close()

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Files indexed:", str(s.file_count))
    table.add_row("Chunks:", str(s.chunk_count))
    table.add_row("Database size:", _human_bytes(s.db_size_bytes))
    table.add_row("Database path:", str(db_path.resolve()))

    console.print(Panel(table, title="[bold]Index stats[/bold]", expand=False))


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


@app.command()
def reset(
    db: Annotated[Optional[Path], typer.Option("--db", help="Database file path.")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Wipe the index (keeps the database file)."""
    db_path = _db_path(db)
    if not db_path.exists():
        err_console.print("[red]Database not found.[/red]")
        raise typer.Exit(1)

    if not yes:
        typer.confirm(
            f"This will delete all indexed data in {db_path}. Continue?",
            abort=True,
        )

    conn = open_db(db_path)
    Repository(conn).reset()
    conn.close()
    console.print("[green]Index reset.[/green]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"
