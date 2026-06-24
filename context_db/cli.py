"""CLI entry-point for context-db.

Commands
--------
ctx init                – Initialise a new database in the current directory.
ctx index <PATH>        – Index (or re-index) a directory tree.
ctx search <QUERY>      – Search the index with BM25 ranking.
ctx stats               – Print index statistics.
ctx reset               – Wipe the index and start fresh.

The database file defaults to ``./context.db`` but can be overridden via the
``--db`` global option or the ``CTX_DB`` environment variable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from context_db.indexer.chunker import Chunker
from context_db.indexer.pipeline import IndexingPipeline, ProgressEvent
from context_db.indexer.scanner import Scanner
from context_db.retrieval.search import SearchEngine
from context_db.storage.db import open_db
from context_db.storage.repository import Repository

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(level: int) -> None:
    """Configure structlog at the given numeric log level.

    Uses the default PrintLoggerFactory (stdout).  The level can be overridden
    per-invocation; structlog's global config is process-wide so this should
    only be called once at startup or in controlled test environments.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
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
console = Console()

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
) -> None:
    """Index a directory tree into the database."""
    if verbose:
        # Re-configure at DEBUG level for this process invocation.
        # In tests, use CTX_LOG_LEVEL=10 env var to avoid global mutation.
        _configure_logging(10)

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
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results (unique files).")] = 10,
    path_filter: Annotated[Optional[str], typer.Option("--path", help="SQL LIKE filter on file path.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
    all_chunks: Annotated[bool, typer.Option("--all-chunks", help="Show all matching chunks, not just the best per file.")] = False,
) -> None:
    """Search the index using BM25 ranking.

    By default each file appears once (its best-matching chunk).
    Use --all-chunks to see every matching chunk from every file.
    """
    db_path = _db_path(db)
    if not db_path.exists():
        console.print("[red]Database not found.[/red] Run [bold]ctx init[/bold] first.")
        raise typer.Exit(1)

    conn = open_db(db_path)
    engine = SearchEngine(conn)
    results = engine.search(
        query,
        limit=limit,
        path_filter=path_filter,
        deduplicate=not all_chunks,
    )
    conn.close()

    if not results:
        console.print(f"[yellow]No results for:[/yellow] {query!r}")
        raise typer.Exit(0)

    if json_output:
        import json
        data = [
            {
                "path": str(r.path),
                "score": r.score,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "preview": r.preview,
            }
            for r in results
        ]
        typer.echo(json.dumps(data, indent=2))
        return

    for i, result in enumerate(results, 1):
        console.rule(
            f"[bold]{i}.[/bold] [cyan]{result.path}[/cyan] "
            f"[dim]lines {result.start_line}–{result.end_line}[/dim] "
            f"[yellow]score={result.score:.4f}[/yellow]"
        )
        console.print(result.preview.strip())
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
        console.print("[red]Database not found.[/red] Run [bold]ctx init[/bold] first.")
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
        console.print("[red]Database not found.[/red]")
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
