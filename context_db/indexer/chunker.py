"""Chunker — splits file content into overlapping fixed-size slices.

Design notes:
* Splitting is character-based (not token-based) for simplicity and speed.
* Chunks are aligned to *line boundaries* so that each chunk starts at the
  beginning of a line and ends at the end of a line.  This preserves
  readability and makes line-range metadata accurate.
* An optional overlap (in lines) can be set to improve retrieval recall at
  chunk boundaries.

The character limit controls the *target* size; actual chunk size may be
slightly larger because lines are never split mid-line.
"""

from __future__ import annotations

import structlog

from context_db.models import Chunk
from pathlib import Path

logger = structlog.get_logger(__name__)

DEFAULT_CHUNK_CHARS: int = 1_500   # ~1 500 chars ≈ 40–60 lines of code
DEFAULT_OVERLAP_LINES: int = 3     # lines repeated between consecutive chunks


class Chunker:
    """Splits file text into :class:`Chunk` objects.

    Parameters
    ----------
    chunk_chars:
        Target maximum number of characters per chunk.  Lines are never split;
        a chunk may exceed this limit if a single line is longer.
    overlap_lines:
        Number of lines from the *end* of a chunk that are repeated at the
        *start* of the next chunk.  Helps retrieval across boundaries.
    """

    def __init__(
        self,
        *,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        overlap_lines: int = DEFAULT_OVERLAP_LINES,
    ) -> None:
        if chunk_chars < 1:
            raise ValueError("chunk_chars must be >= 1")
        if overlap_lines < 0:
            raise ValueError("overlap_lines must be >= 0")
        self._chunk_chars = chunk_chars
        self._overlap_lines = overlap_lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_file(self, path: Path, content: str) -> list[Chunk]:
        """Split *content* (already read from *path*) into :class:`Chunk` objects.

        Parameters
        ----------
        path:
            The source file path stored in each chunk's metadata.
        content:
            Full text content of the file.

        Returns
        -------
        list[Chunk]
            Empty list if *content* is empty.
        """
        if not content:
            return []

        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        chunks: list[Chunk] = []
        line_idx = 0          # 0-based index into *lines*
        total_lines = len(lines)

        while line_idx < total_lines:
            chunk_lines: list[str] = []
            char_count = 0
            start_idx = line_idx

            # Accumulate lines until we hit the character limit.
            while line_idx < total_lines:
                line = lines[line_idx]
                # Always include at least one line to prevent infinite loops.
                if chunk_lines and char_count + len(line) > self._chunk_chars:
                    break
                chunk_lines.append(line)
                char_count += len(line)
                line_idx += 1

            # 1-based line numbers for the output.
            start_line = start_idx + 1
            end_line = start_idx + len(chunk_lines)
            chunk_content = "".join(chunk_lines).rstrip("\n")

            chunks.append(
                Chunk(
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content,
                )
            )

            # Apply overlap: step back N lines so the next chunk re-reads them.
            if self._overlap_lines > 0 and line_idx < total_lines:
                line_idx = max(start_idx + 1, line_idx - self._overlap_lines)

        logger.debug(
            "chunk_file",
            path=str(path),
            lines=total_lines,
            chunks=len(chunks),
            chunk_chars=self._chunk_chars,
        )
        return chunks

    def chunk_files(self, file_contents: list[tuple[Path, str]]) -> list[Chunk]:
        """Convenience wrapper: chunk multiple *(path, content)* pairs."""
        all_chunks: list[Chunk] = []
        for path, content in file_contents:
            all_chunks.extend(self.chunk_file(path, content))
        return all_chunks
