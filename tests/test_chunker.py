"""Tests for context_db.indexer.chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from context_db.indexer.chunker import Chunker
from context_db.models import Chunk


class TestChunker:
    def test_empty_content_returns_empty_list(self, tmp_path: Path) -> None:
        chunker = Chunker()
        result = chunker.chunk_file(tmp_path / "f.py", "")
        assert result == []

    def test_single_chunk_for_small_file(self, tmp_path: Path) -> None:
        content = "line 1\nline 2\nline 3\n"
        chunker = Chunker(chunk_chars=10_000)
        chunks = chunker.chunk_file(tmp_path / "f.py", content)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3

    def test_multiple_chunks_for_large_file(self, tmp_path: Path) -> None:
        lines = [f"line {i}\n" for i in range(1, 101)]
        content = "".join(lines)
        chunker = Chunker(chunk_chars=100, overlap_lines=0)
        chunks = chunker.chunk_file(tmp_path / "f.py", content)
        assert len(chunks) > 1

    def test_chunk_line_ranges_are_contiguous_without_overlap(self, tmp_path: Path) -> None:
        lines = [f"line {i}\n" for i in range(1, 51)]
        content = "".join(lines)
        chunker = Chunker(chunk_chars=50, overlap_lines=0)
        chunks = chunker.chunk_file(tmp_path / "f.py", content)
        # Verify no line numbers are skipped.
        all_lines = set()
        for c in chunks:
            all_lines.update(range(c.start_line, c.end_line + 1))
        expected = set(range(1, 51))
        assert expected == all_lines

    def test_overlap_repeats_lines(self, tmp_path: Path) -> None:
        lines = [f"line {i}\n" for i in range(1, 31)]
        content = "".join(lines)
        chunker = Chunker(chunk_chars=60, overlap_lines=3)
        chunks = chunker.chunk_file(tmp_path / "f.py", content)
        if len(chunks) >= 2:
            # The second chunk should start before the first ended.
            assert chunks[1].start_line <= chunks[0].end_line

    def test_chunk_preserves_path(self, tmp_path: Path) -> None:
        path = tmp_path / "src" / "module.py"
        chunker = Chunker()
        chunks = chunker.chunk_file(path, "x = 1\n")
        assert chunks[0].path == path

    def test_chunk_models_are_valid(self, tmp_path: Path) -> None:
        content = "a\nb\nc\nd\ne\n"
        chunker = Chunker(chunk_chars=10, overlap_lines=0)
        chunks = chunker.chunk_file(tmp_path / "f.py", content)
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.start_line >= 1
            assert c.end_line >= c.start_line
            assert c.content

    def test_single_very_long_line(self, tmp_path: Path) -> None:
        long_line = "x" * 5000 + "\n"
        chunker = Chunker(chunk_chars=100, overlap_lines=0)
        chunks = chunker.chunk_file(tmp_path / "f.py", long_line)
        # Long line must still produce at least one chunk.
        assert len(chunks) >= 1

    def test_invalid_chunk_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_chars"):
            Chunker(chunk_chars=0)

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap_lines"):
            Chunker(overlap_lines=-1)

    def test_chunk_files_convenience(self, tmp_path: Path) -> None:
        pairs = [
            (tmp_path / "a.py", "import os\n"),
            (tmp_path / "b.py", "import sys\n"),
        ]
        chunker = Chunker()
        chunks = chunker.chunk_files(pairs)
        assert len(chunks) == 2

    def test_no_trailing_newline_in_content(self, tmp_path: Path) -> None:
        content = "line1\nline2"  # no trailing newline
        chunker = Chunker(chunk_chars=1000)
        chunks = chunker.chunk_file(tmp_path / "f.py", content)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
