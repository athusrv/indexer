"""Tests for context_db.indexer.fingerprint."""

from __future__ import annotations

from pathlib import Path

import pytest

from context_db.indexer.fingerprint import Fingerprinter, fingerprint_file, hash_file
from context_db.models import DiscoveredFile, FileMetadata


class TestHashFile:
    def test_returns_64_char_hex(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hello")
        digest = hash_file(f)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("same content")
        b.write_text("same content")
        assert hash_file(a) == hash_file(b)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("content A")
        b.write_text("content B")
        assert hash_file(a) != hash_file(b)

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        digest = hash_file(f)
        # SHA-256 of empty string is well-known
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class TestFingerprintFile:
    def test_returns_file_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("x = 1")
        meta = fingerprint_file(f)
        assert isinstance(meta, FileMetadata)
        assert meta.path == f
        assert len(meta.hash) == 64
        assert meta.mtime > 0
        assert meta.size == f.stat().st_size


class TestFingerprinter:
    def _make_files(self, tmp_path: Path) -> list[DiscoveredFile]:
        files = []
        for name, content in [("a.py", "a=1"), ("b.py", "b=2")]:
            p = tmp_path / name
            p.write_text(content)
            files.append(DiscoveredFile(path=p))
        return files

    def test_compute_returns_metadata_list(self, tmp_path: Path) -> None:
        discovered = self._make_files(tmp_path)
        fp = Fingerprinter()
        results = fp.compute(discovered)
        assert len(results) == 2
        assert all(isinstance(m, FileMetadata) for m in results)

    def test_diff_detects_new_files(self, tmp_path: Path) -> None:
        discovered = self._make_files(tmp_path)
        fp = Fingerprinter(stored_metadata={})
        current = fp.compute(discovered)
        changeset = fp.diff(current)
        assert len(changeset.new_files) == 2
        assert len(changeset.modified_files) == 0
        assert len(changeset.deleted_paths) == 0

    def test_diff_detects_no_change(self, tmp_path: Path) -> None:
        discovered = self._make_files(tmp_path)
        fp = Fingerprinter()
        current = fp.compute(discovered)
        # Pre-populate stored state with the same hashes.
        stored = {str(m.path): m for m in current}
        fp2 = Fingerprinter(stored_metadata=stored)
        changeset = fp2.diff(current)
        assert not changeset.has_changes

    def test_diff_detects_modified_file(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("original")
        old_meta = fingerprint_file(f)
        stored = {str(f): old_meta}

        # Now "modify" the file.
        f.write_text("changed content")
        new_meta = fingerprint_file(f)

        fp = Fingerprinter(stored_metadata=stored)
        changeset = fp.diff([new_meta])
        assert len(changeset.modified_files) == 1
        assert changeset.modified_files[0].path == f

    def test_diff_detects_deleted_files(self, tmp_path: Path) -> None:
        f = tmp_path / "gone.py"
        f.write_text("x")
        old_meta = fingerprint_file(f)
        stored = {str(f): old_meta}

        # File is no longer on disk — so *current* is empty.
        fp = Fingerprinter(stored_metadata=stored)
        changeset = fp.diff([])
        assert len(changeset.deleted_paths) == 1
        assert changeset.deleted_paths[0] == f

    def test_changeset_total_changes(self, tmp_path: Path) -> None:
        f = tmp_path / "x.py"
        f.write_text("new")
        new_meta = fingerprint_file(f)
        fp = Fingerprinter()
        changeset = fp.diff([new_meta])
        assert changeset.total_changes == 1
        assert changeset.has_changes
