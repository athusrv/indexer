"""Tests for context_db.indexer.scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from context_db.indexer.scanner import Scanner


class TestScanner:
    def test_scan_finds_text_files(self, sample_tree: Path) -> None:
        scanner = Scanner()
        found = scanner.scan(sample_tree)
        paths = {f.path.name for f in found}
        assert "auth.py" in paths
        assert "utils.py" in paths
        assert "README.md" in paths

    def test_scan_ignores_node_modules(self, sample_tree: Path) -> None:
        scanner = Scanner()
        found = scanner.scan(sample_tree)
        # Check that no discovered file lives *inside* a node_modules directory.
        assert not any(
            "node_modules" in f.path.parts for f in found
        )

    def test_scan_ignores_pycache(self, sample_tree: Path) -> None:
        scanner = Scanner()
        found = scanner.scan(sample_tree)
        assert not any("__pycache__" in f.path.parts for f in found)

    def test_scan_ignores_binary_extension(self, sample_tree: Path) -> None:
        scanner = Scanner()
        found = scanner.scan(sample_tree)
        paths = {f.path.name for f in found}
        assert "cached.pyc" not in paths

    def test_scan_custom_ignore_pattern(self, sample_tree: Path) -> None:
        scanner = Scanner(ignore_patterns=["*.md"])
        found = scanner.scan(sample_tree)
        paths = {f.path.name for f in found}
        assert "README.md" not in paths

    def test_scan_returns_discovered_file_objects(self, sample_tree: Path) -> None:
        from context_db.models import DiscoveredFile
        scanner = Scanner()
        found = scanner.scan(sample_tree)
        assert all(isinstance(f, DiscoveredFile) for f in found)

    def test_scan_raises_on_nonexistent_path(self, tmp_path: Path) -> None:
        scanner = Scanner()
        with pytest.raises(ValueError, match="does not exist"):
            scanner.scan(tmp_path / "nonexistent")

    def test_scan_raises_on_file_not_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        scanner = Scanner()
        with pytest.raises(ValueError, match="not a directory"):
            scanner.scan(f)

    def test_scan_incremental_returns_only_newer(self, tmp_path: Path) -> None:
        f = tmp_path / "old.py"
        f.write_text("x = 1")
        # Set mtime to the past
        import os, time
        old_mtime = time.time() - 1000
        os.utime(f, (old_mtime, old_mtime))

        new_file = tmp_path / "new.py"
        new_file.write_text("y = 2")

        scanner = Scanner()
        results = scanner.scan_incremental(tmp_path, since_mtime=old_mtime + 1)
        names = {r.path.name for r in results}
        assert "new.py" in names
        assert "old.py" not in names

    def test_scan_skips_binary_content(self, tmp_path: Path) -> None:
        binary = tmp_path / "data.dat"
        binary.write_bytes(b"\x00\x01\x02\x03")
        scanner = Scanner()
        found = scanner.scan(tmp_path)
        assert not any(f.path.name == "data.dat" for f in found)

    def test_scan_custom_ignore_dirs(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "my_ignore"
        custom_dir.mkdir()
        (custom_dir / "file.py").write_text("x = 1")
        scanner = Scanner(ignore_dirs=["my_ignore"])
        found = scanner.scan(tmp_path)
        assert not any("my_ignore" in str(f.path) for f in found)
