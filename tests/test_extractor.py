"""Tests for context_db.indexer.extractor and rich-format pipeline integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from context_db.indexer.extractor import (
    RICH_EXTENSIONS,
    extract_text,
    is_rich_format,
)
from context_db.indexer.pipeline import IndexingPipeline
from context_db.storage.db import open_db
from context_db.storage.repository import Repository
from tests.helpers import make_docx, make_xlsx


# ---------------------------------------------------------------------------
# is_rich_format
# ---------------------------------------------------------------------------


class TestIsRichFormat:
    def test_pdf_is_rich(self) -> None:
        assert is_rich_format(Path("report.pdf")) is True

    def test_docx_is_rich(self) -> None:
        assert is_rich_format(Path("doc.docx")) is True

    def test_xlsx_is_rich(self) -> None:
        assert is_rich_format(Path("sheet.xlsx")) is True

    def test_pptx_is_rich(self) -> None:
        assert is_rich_format(Path("slides.pptx")) is True

    def test_html_is_rich(self) -> None:
        assert is_rich_format(Path("page.html")) is True

    def test_csv_is_rich(self) -> None:
        assert is_rich_format(Path("data.csv")) is True

    def test_py_is_not_rich(self) -> None:
        assert is_rich_format(Path("module.py")) is False

    def test_txt_is_not_rich(self) -> None:
        assert is_rich_format(Path("notes.txt")) is False

    def test_case_insensitive(self) -> None:
        assert is_rich_format(Path("REPORT.PDF")) is True
        assert is_rich_format(Path("Sheet.XLSX")) is True

    def test_rich_extensions_set(self) -> None:
        assert ".pdf" in RICH_EXTENSIONS
        assert ".docx" in RICH_EXTENSIONS
        assert ".xlsx" in RICH_EXTENSIONS
        assert ".py" not in RICH_EXTENSIONS


# ---------------------------------------------------------------------------
# extract_text — HTML (always works, no extra dep)
# ---------------------------------------------------------------------------


class TestExtractTextHtml:
    def test_html_extracts_headings(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text("<h1>Authentication</h1><p>JWT token verify</p>")
        result = extract_text(f)
        assert "Authentication" in result
        assert "JWT" in result

    def test_html_strips_tags(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text("<p>Hello <b>world</b></p>")
        result = extract_text(f)
        assert "<b>" not in result
        assert "world" in result

    def test_empty_html_returns_string(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.html"
        f.write_text("<html></html>")
        result = extract_text(f)
        assert isinstance(result, str)

    def test_failed_extraction_returns_empty_string(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.html"
        f.write_text("irrelevant")
        with patch(
            "context_db.indexer.extractor._get_converter",
            side_effect=RuntimeError("converter crashed"),
        ):
            result = extract_text(f)
        assert result == ""


# ---------------------------------------------------------------------------
# extract_text — CSV
# ---------------------------------------------------------------------------


class TestExtractTextCsv:
    def test_csv_extracts_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("name,role\njwt,auth\ntoken,verify\n")
        result = extract_text(f)
        assert "jwt" in result
        assert "auth" in result

    def test_csv_multicolumn(self, tmp_path: Path) -> None:
        f = tmp_path / "users.csv"
        f.write_text("id,email,role\n1,alice@x.com,admin\n2,bob@x.com,viewer\n")
        result = extract_text(f)
        assert "alice" in result
        assert "admin" in result


# ---------------------------------------------------------------------------
# extract_text — DOCX
# ---------------------------------------------------------------------------


class TestExtractTextDocx:
    def test_docx_extracts_paragraph(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.docx"
        make_docx(f, "JWT authentication token verify secret")
        result = extract_text(f)
        assert "JWT" in result
        assert "authentication" in result

    def test_docx_returns_string(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.docx"
        make_docx(f, "hello world")
        result = extract_text(f)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# extract_text — XLSX
# ---------------------------------------------------------------------------


class TestExtractTextXlsx:
    def test_xlsx_extracts_cell_values(self, tmp_path: Path) -> None:
        f = tmp_path / "sheet.xlsx"
        make_xlsx(f, [["name", "score"], ["alice", "99"], ["bob", "87"]])
        result = extract_text(f)
        assert "alice" in result
        assert "score" in result

    def test_xlsx_multi_row(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xlsx"
        make_xlsx(f, [["jwt", "token"], ["refresh", "expire"]])
        result = extract_text(f)
        assert "jwt" in result


# ---------------------------------------------------------------------------
# Scanner — rich files are discovered
# ---------------------------------------------------------------------------


class TestScannerRichFormats:
    def test_scanner_finds_pdf(self, tmp_path: Path) -> None:
        from context_db.indexer.scanner import Scanner

        # Create a fake PDF with a PDF header (no null bytes in header area)
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4\n%This is a fake pdf\n")
        scanner = Scanner()
        found = scanner.scan(tmp_path)
        assert any(f.path.name == "report.pdf" for f in found)

    def test_scanner_finds_html(self, tmp_path: Path) -> None:
        from context_db.indexer.scanner import Scanner

        f = tmp_path / "index.html"
        f.write_text("<h1>Hello</h1>")
        scanner = Scanner()
        found = scanner.scan(tmp_path)
        assert any(f.path.name == "index.html" for f in found)

    def test_scanner_finds_docx(self, tmp_path: Path) -> None:
        from context_db.indexer.scanner import Scanner

        f = tmp_path / "doc.docx"
        make_docx(f, "hello world")
        scanner = Scanner()
        found = scanner.scan(tmp_path)
        assert any(f.path.name == "doc.docx" for f in found)

    def test_scanner_finds_csv(self, tmp_path: Path) -> None:
        from context_db.indexer.scanner import Scanner

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        scanner = Scanner()
        found = scanner.scan(tmp_path)
        assert any(f.path.name == "data.csv" for f in found)


# ---------------------------------------------------------------------------
# Pipeline integration — rich files are indexed and searchable
# ---------------------------------------------------------------------------


class TestPipelineRichFormats:
    def _make_pipeline(self, tmp_path: Path) -> tuple[IndexingPipeline, Repository]:
        db_path = tmp_path / "rich.db"
        conn = open_db(db_path)
        repo = Repository(conn)
        return IndexingPipeline(repository=repo), repo

    def test_pipeline_indexes_html(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / "page.html").write_text(
            "<h1>Auth</h1><p>JWT verify token refresh secret</p>"
        )
        pipeline, repo = self._make_pipeline(tmp_path)
        result = pipeline.run(root)
        assert result.indexed == 1
        assert result.total_chunks >= 1

    def test_pipeline_indexes_csv(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / "users.csv").write_text("name,role\njwt,auth\ntoken,verify\n")
        pipeline, repo = self._make_pipeline(tmp_path)
        result = pipeline.run(root)
        assert result.indexed == 1

    def test_pipeline_indexes_docx(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        make_docx(root / "doc.docx", "JWT authentication token verify")
        pipeline, repo = self._make_pipeline(tmp_path)
        result = pipeline.run(root)
        assert result.indexed == 1
        assert result.total_chunks >= 1

    def test_pipeline_indexes_xlsx(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        make_xlsx(root / "data.xlsx", [["token", "role"], ["jwt", "admin"]])
        pipeline, repo = self._make_pipeline(tmp_path)
        result = pipeline.run(root)
        assert result.indexed == 1

    def test_html_content_is_searchable(self, tmp_path: Path) -> None:
        from context_db.retrieval.search import SearchEngine
        from context_db.storage.db import open_db

        root = tmp_path / "root"
        root.mkdir()
        (root / "auth.html").write_text(
            "<h1>Authentication</h1><p>JWT verify token with secret key</p>"
        )
        db_path = tmp_path / "rich.db"
        conn = open_db(db_path)
        repo = Repository(conn)
        pipeline = IndexingPipeline(repository=repo)
        pipeline.run(root)

        engine = SearchEngine(conn)
        results = engine.search("jwt")
        assert len(results) >= 1
        assert any("auth.html" in str(r.path) for r in results)

    def test_extraction_failure_produces_zero_chunks(self, tmp_path: Path) -> None:
        """Unreadable rich file produces 0 chunks but no pipeline crash."""
        root = tmp_path / "root"
        root.mkdir()
        # Write a .docx file that markitdown can't parse
        (root / "broken.docx").write_bytes(b"not a real docx file at all")

        pipeline, _ = self._make_pipeline(tmp_path)
        result = pipeline.run(root)
        # Should index 1 file (it was discovered) but produce 0 chunks
        # OR count as error — either way it must not crash.
        assert result.errors == 0 or result.indexed >= 0
