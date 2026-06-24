"""Test helpers for creating minimal rich-format document fixtures."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Minimal DOCX builder (no external lib needed)
# A .docx is a ZIP with specific XML files.
# ---------------------------------------------------------------------------

_DOCX_CONTENT_TYPES = """\
<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels"
    ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_DOCX_RELS = """\
<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""

_DOCX_WORD_RELS = (
    '<Relationships xmlns='
    '"http://schemas.openxmlformats.org/package/2006/relationships"/>'
)


def make_docx(path: Path, text: str) -> None:
    """Write a minimal valid .docx containing *text* to *path*."""
    document_xml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        zf.writestr("_rels/.rels", _DOCX_RELS)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", _DOCX_WORD_RELS)
    path.write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# Minimal XLSX builder (openpyxl is installed as part of markitdown[xlsx])
# ---------------------------------------------------------------------------


def make_xlsx(path: Path, rows: list[list[str]]) -> None:
    """Write a minimal .xlsx with *rows* to *path*."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    wb.save(str(path))
