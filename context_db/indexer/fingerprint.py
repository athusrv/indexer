"""Fingerprinter — hashes files and detects new / modified / deleted items.

Design goals:
* SHA-256 is computed in streaming fashion to handle large files without
  loading them fully into memory.
* Change detection is done by comparing against the stored metadata already
  in the repository, so no extra state file is needed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

from context_db.models import ChangeSet, DiscoveredFile, FileMetadata

logger = structlog.get_logger(__name__)

_CHUNK_SIZE = 65_536  # 64 KiB read buffer


def hash_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> FileMetadata:
    """Compute the :class:`FileMetadata` for a single file.

    Raises :class:`OSError` if the file cannot be read.
    """
    stat = path.stat()
    file_hash = hash_file(path)
    return FileMetadata(
        path=path,
        hash=file_hash,
        mtime=stat.st_mtime,
        size=stat.st_size,
    )


class Fingerprinter:
    """Computes :class:`FileMetadata` and detects changes against stored state.

    Parameters
    ----------
    stored_metadata:
        A mapping of *absolute path string* → :class:`FileMetadata` that
        represents the currently indexed state.  Provided by the storage layer.
    """

    def __init__(self, stored_metadata: dict[str, FileMetadata] | None = None) -> None:
        self._stored: dict[str, FileMetadata] = stored_metadata or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, discovered: list[DiscoveredFile]) -> list[FileMetadata]:
        """Hash all *discovered* files and return their metadata."""
        results: list[FileMetadata] = []
        for df in discovered:
            try:
                meta = fingerprint_file(df.path)
                results.append(meta)
            except OSError as exc:
                logger.warning("fingerprint_error", path=str(df.path), error=str(exc))
        logger.debug("fingerprint_complete", count=len(results))
        return results

    def diff(self, current: list[FileMetadata]) -> ChangeSet:
        """Compute the delta between *current* metadata and the stored state.

        Returns
        -------
        ChangeSet
            Contains lists of new files, modified files, and deleted paths.
        """
        current_by_path = {str(m.path): m for m in current}

        new_files: list[FileMetadata] = []
        modified_files: list[FileMetadata] = []

        for path_str, meta in current_by_path.items():
            if path_str not in self._stored:
                new_files.append(meta)
            elif self._stored[path_str].hash != meta.hash:
                modified_files.append(meta)

        deleted_paths: list[Path] = [
            Path(p) for p in self._stored if p not in current_by_path
        ]

        changeset = ChangeSet(
            new_files=new_files,
            modified_files=modified_files,
            deleted_paths=deleted_paths,
        )

        logger.debug(
            "diff_complete",
            new=len(new_files),
            modified=len(modified_files),
            deleted=len(deleted_paths),
        )
        return changeset
