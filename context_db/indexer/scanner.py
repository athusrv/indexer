"""Filesystem scanner — recursively discovers indexable source files.

Skips non-text files (binaries), honours configurable ignore patterns, and
supports incremental mode by optionally returning only files modified since a
given timestamp.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

import structlog

from context_db.indexer.extractor import RICH_EXTENSIONS
from context_db.models import DiscoveredFile

logger = structlog.get_logger(__name__)

# Directories that are always skipped regardless of user configuration.
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "dist",
        "build",
        ".next",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "venv",
        ".venv",
        "env",
        ".tox",
        "coverage",
        ".coverage",
        "htmlcov",
    }
)

# File extensions treated as binary/non-indexable by default.
DEFAULT_IGNORE_EXTENSIONS: frozenset[str] = frozenset(
    {
        # compiled / binary
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        ".obj",
        ".o",
        ".a",
        ".lib",
        # archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # media
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".webp",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        # fonts
        ".ttf",
        ".woff",
        ".woff2",
        ".eot",
        # data / lock
        ".db",
        ".sqlite",
        ".sqlite3",
        ".lock",
        # misc binary
        ".pkl",
        ".npy",
        ".npz",
        # NOTE: .pdf, .docx, .xlsx, .pptx, .html, .csv, .xml are intentionally
        # NOT in this list — they are handled by the rich-format extractor.
    }
)

# Maximum file size to index (16 MB). Larger files are skipped.
MAX_FILE_BYTES: int = 16 * 1024 * 1024


class Scanner:
    """Recursively scans a directory and yields :class:`DiscoveredFile` objects.

    Parameters
    ----------
    ignore_dirs:
        Additional directory names to skip on top of :data:`DEFAULT_IGNORE_DIRS`.
    ignore_patterns:
        Shell-style glob patterns (fnmatch) matched against the *relative* path
        of each file (e.g. ``"*.min.js"``).
    max_file_bytes:
        Files larger than this limit are silently skipped.
    """

    def __init__(
        self,
        *,
        ignore_dirs: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_file_bytes: int = MAX_FILE_BYTES,
    ) -> None:
        self._ignore_dirs = DEFAULT_IGNORE_DIRS | frozenset(ignore_dirs or [])
        self._ignore_patterns = list(ignore_patterns or [])
        self._max_file_bytes = max_file_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, root: Path) -> list[DiscoveredFile]:
        """Return all indexable files under *root*.

        Parameters
        ----------
        root:
            The directory to scan.  Raises :class:`ValueError` if the path is
            not an existing directory.
        """
        root = root.resolve()
        if not root.exists():
            raise ValueError(f"Path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        discovered: list[DiscoveredFile] = []
        skipped_dirs = 0
        skipped_files = 0

        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            current = Path(dirpath)

            # Prune ignored directories *in-place* so os.walk won't descend.
            before = len(dirnames)
            dirnames[:] = [
                d for d in dirnames if not self._should_ignore_dir(d, current, root)
            ]
            skipped_dirs += before - len(dirnames)

            for filename in filenames:
                filepath = current / filename
                rel = filepath.relative_to(root)

                if self._should_ignore_file(filepath, rel):
                    skipped_files += 1
                    continue

                discovered.append(DiscoveredFile(path=filepath))

        logger.debug(
            "scan_complete",
            root=str(root),
            discovered=len(discovered),
            skipped_dirs=skipped_dirs,
            skipped_files=skipped_files,
        )
        return discovered

    def scan_incremental(self, root: Path, since_mtime: float) -> list[DiscoveredFile]:
        """Return only files whose ``mtime`` is newer than *since_mtime*.

        Useful for quick re-scans after an initial full index.
        """
        all_files = self.scan(root)
        return [f for f in all_files if f.path.stat().st_mtime > since_mtime]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_ignore_dir(self, dirname: str, parent: Path, root: Path) -> bool:
        if dirname in self._ignore_dirs:
            return True
        # Also allow pattern-matching against the relative dir path.
        try:
            rel = (parent / dirname).relative_to(root)
        except ValueError:
            return False
        rel_str = str(rel)
        return any(fnmatch.fnmatch(rel_str, pat) for pat in self._ignore_patterns)

    def _should_ignore_file(self, filepath: Path, rel: Path) -> bool:
        # Skip by extension.
        if filepath.suffix.lower() in DEFAULT_IGNORE_EXTENSIONS:
            return True

        # Skip by size.
        try:
            if filepath.stat().st_size > self._max_file_bytes:
                return True
        except OSError:
            return True

        # Skip non-readable files.
        if not os.access(filepath, os.R_OK):
            return True

        # Skip symlinks pointing outside the tree.
        if filepath.is_symlink():
            try:
                filepath.resolve().relative_to(filepath.parent.resolve())
            except ValueError:
                return True

        # Apply user glob patterns against the relative path string.
        rel_str = str(rel)
        if any(fnmatch.fnmatch(rel_str, pat) for pat in self._ignore_patterns):
            return True

        # Rich-format documents (.pdf, .docx, .xlsx, …) are binary but handled
        # by the extractor — skip the null-byte sniff for them.
        if filepath.suffix.lower() in RICH_EXTENSIONS:
            return False

        # Quick binary-content sniff: read first 512 bytes and check for NUL.
        try:
            with open(filepath, "rb") as fh:
                sample = fh.read(512)
            if b"\x00" in sample:
                return True
        except OSError:
            return True

        return False
