"""Indexing pipeline: scan → fingerprint → read → chunk → persist → embed.

The pipeline is intentionally stateless: all state lives in SQLite.  Every
run determines the minimal set of files to (re-)index by comparing on-disk
fingerprints against the stored metadata.

Embedding step (optional)
--------------------------
When an :class:`~context_db.embeddings.embedder.Embedder` is injected the
pipeline adds a step after FTS indexing:

* Chunks that already have an embedding for the active model are skipped.
* New and modified chunks (whose old embeddings were cascade-deleted) are
  embedded in a single batch call.
* ``reembed=True`` first wipes all stored vectors for the model so every
  chunk is re-embedded from scratch.

Progress reporting
------------------
The pipeline accepts an optional ``progress_callback`` that is called with
a :class:`ProgressEvent` after each file is processed.  The CLI uses this to
render a Rich progress bar; other callers can ignore it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import structlog

from context_db.indexer.chunker import Chunker
from context_db.indexer.extractor import extract_text, is_rich_format
from context_db.indexer.fingerprint import Fingerprinter
from context_db.indexer.scanner import Scanner
from context_db.models import ChangeSet, FileMetadata, IndexStats
from context_db.storage.repository import Repository

if TYPE_CHECKING:
    from context_db.embeddings.embedder import Embedder

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProgressEvent:
    """Emitted by the pipeline after each file is processed."""

    current: int
    total: int
    path: Path
    action: str  # "index" | "delete" | "skip"

    @property
    def percent(self) -> float:
        return (self.current / self.total * 100) if self.total else 0.0


ProgressCallback = Callable[[ProgressEvent], None]


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Summary of a completed pipeline run."""

    indexed: int = 0
    deleted: int = 0
    skipped: int = 0
    errors: int = 0
    duration_s: float = 0.0
    total_chunks: int = 0
    embedded: int = 0  # chunks that had embeddings generated this run
    changeset: ChangeSet = field(
        default_factory=lambda: ChangeSet(new_files=[], modified_files=[], deleted_paths=[])
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IndexingPipeline:
    """Orchestrates the full indexing flow.

    Parameters
    ----------
    repository:
        The storage repository to read from and write to.
    scanner:
        Scanner instance (caller can inject custom ignore patterns).
    chunker:
        Chunker instance (caller can control chunk size / overlap).
    progress_callback:
        Optional callable invoked after each file-level action.
    """

    def __init__(
        self,
        repository: Repository,
        scanner: Scanner | None = None,
        chunker: Chunker | None = None,
        progress_callback: ProgressCallback | None = None,
        embedder: Embedder | None = None,
        reembed: bool = False,
    ) -> None:
        self._repo = repository
        self._scanner = scanner or Scanner()
        self._chunker = chunker or Chunker()
        self._progress_cb = progress_callback
        self._embedder = embedder
        self._reembed = reembed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, root: Path) -> RunResult:
        """Index the directory tree rooted at *root*.

        Steps
        -----
        1. Scan the filesystem for candidate files.
        2. Load stored metadata from the DB.
        3. Fingerprint all discovered files.
        4. Compute the changeset (new / modified / deleted).
        5. For each changed file: read → chunk → persist.
        6. Delete stale file records.
        7. Optimise the FTS index.
        """
        t0 = time.perf_counter()
        result = RunResult()

        # ── 1. Scan ───────────────────────────────────────────────────
        logger.info("pipeline_scan_start", root=str(root))
        discovered = self._scanner.scan(root)
        logger.info("pipeline_scan_done", discovered=len(discovered))

        # ── 2. Load stored state ──────────────────────────────────────
        stored = self._repo.get_all_file_metadata()

        # ── 3. Fingerprint ────────────────────────────────────────────
        fingerprinter = Fingerprinter(stored_metadata=stored)
        current_metas = fingerprinter.compute(discovered)

        # ── 4. Diff ───────────────────────────────────────────────────
        changeset = fingerprinter.diff(current_metas)
        result.changeset = changeset

        to_index: list[FileMetadata] = changeset.new_files + changeset.modified_files
        total_actions = len(to_index) + len(changeset.deleted_paths)
        action_no = 0

        logger.info(
            "pipeline_changeset",
            new=len(changeset.new_files),
            modified=len(changeset.modified_files),
            deleted=len(changeset.deleted_paths),
        )

        # ── 5. Index changed files ────────────────────────────────────
        for meta in to_index:
            action_no += 1
            try:
                content = self._read_file(meta.path)
                chunks = self._chunker.chunk_file(meta.path, content)
                self._repo.replace_file_chunks(meta, chunks)
                result.indexed += 1
                result.total_chunks += len(chunks)
                logger.debug("file_indexed", path=str(meta.path), chunks=len(chunks))
            except Exception as exc:
                result.errors += 1
                logger.error("file_index_error", path=str(meta.path), error=str(exc))

            self._emit(action_no, total_actions, meta.path, "index")

        # ── 6. Delete stale files ─────────────────────────────────────
        for path in changeset.deleted_paths:
            action_no += 1
            try:
                self._repo.delete_file_transactional(path)
                result.deleted += 1
                logger.debug("file_deleted", path=str(path))
            except Exception as exc:
                result.errors += 1
                logger.error("file_delete_error", path=str(path), error=str(exc))

            self._emit(action_no, total_actions, path, "delete")

        result.skipped = len(discovered) - len(to_index)

        # ── 7. Optimise FTS ───────────────────────────────────────────
        if result.indexed or result.deleted:
            self._repo.optimize_fts()

        # ── 8. Generate embeddings (optional) ────────────────────────
        if self._embedder is not None:
            result.embedded = self._embed_step()

        result.duration_s = time.perf_counter() - t0
        logger.info(
            "pipeline_done",
            indexed=result.indexed,
            deleted=result.deleted,
            skipped=result.skipped,
            errors=result.errors,
            chunks=result.total_chunks,
            duration_s=round(result.duration_s, 3),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> str:
        """Return the text content of *path*.

        Rich-format documents (PDF, DOCX, XLSX, …) are converted to Markdown
        via the extractor.  Everything else is read as UTF-8 with a latin-1
        fallback for files that contain non-UTF-8 bytes.
        """
        if is_rich_format(path):
            return extract_text(path)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    def _embed_step(self) -> int:
        """Embed all chunks that lack a vector for the active model.

        When ``reembed`` is ``True``, existing vectors are deleted first so
        every chunk is re-embedded from scratch.

        Returns the number of chunks embedded (0 if already up-to-date).
        """
        assert self._embedder is not None
        model = self._embedder.model_name

        if self._reembed:
            logger.info("pipeline_reembed", model=model)
            self._repo.delete_embeddings_for_model(model)

        missing = self._repo.get_chunks_without_embeddings(model)
        if not missing:
            logger.debug("pipeline_embed_skip", reason="all_up_to_date", model=model)
            return 0

        logger.info("pipeline_embed_start", count=len(missing), model=model)
        contents = [c.content for c in missing]
        chunk_ids = [c.id for c in missing]
        try:
            vectors = self._embedder.embed_batch(contents)
            self._repo.upsert_embeddings_batch(chunk_ids, vectors, model)
            logger.info("pipeline_embed_done", embedded=len(missing), model=model)
            return len(missing)
        except Exception as exc:
            logger.error("pipeline_embed_error", error=str(exc), model=model)
            return 0

    def _emit(
        self, current: int, total: int, path: Path, action: str
    ) -> None:
        if self._progress_cb is not None:
            try:
                self._progress_cb(
                    ProgressEvent(
                        current=current,
                        total=total,
                        path=path,
                        action=action,
                    )
                )
            except Exception:
                pass  # never let a progress callback crash the pipeline
