"""Rich-format text extractor — converts documents to plain text for chunking.

Supported formats (via markitdown)
------------------------------------
PDF      (.pdf)          — text layer extraction; falls back to OCR if needed
Word     (.docx, .doc)   — paragraph + table text
Excel    (.xlsx, .xls)   — sheet names, cell values
PowerPoint (.pptx)       — slide text
HTML     (.html, .htm)   — stripped markup
CSV      (.csv)          — rows as text
XML      (.xml)          — element content

Plain text formats (all other indexable files) are handled by the caller's
normal UTF-8 / latin-1 read path and never go through this module.

Design notes
------------
* A single ``MarkItDown`` instance is reused for all conversions (construction
  is relatively expensive due to magika model loading).
* All extraction errors are caught and logged; the function returns an empty
  string so the pipeline produces zero chunks for an unreadable document
  rather than crashing.
* The extracted text is Markdown — markitdown preserves headings, bullet
  points, and table structure, which is useful context for retrieval.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Extensions handled by this extractor (NOT plain-text reads)
# ---------------------------------------------------------------------------

RICH_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".html",
        ".htm",
        ".csv",
        ".xml",
    }
)


@lru_cache(maxsize=1)
def _get_converter():
    """Return a cached MarkItDown instance (lazy, built on first use)."""
    from markitdown import MarkItDown  # local import keeps startup fast

    return MarkItDown()


def extract_text(path: Path) -> str:
    """Convert a rich-format document to plain Markdown text.

    Parameters
    ----------
    path:
        Absolute path to the document.

    Returns
    -------
    str
        Extracted text (Markdown-flavoured).  Empty string if extraction fails
        or the file produces no text content.
    """
    try:
        converter = _get_converter()
        result = converter.convert(str(path))
        text = result.text_content or ""
        logger.debug(
            "extract_text",
            path=str(path),
            suffix=path.suffix,
            chars=len(text),
        )
        return text
    except Exception as exc:
        logger.warning(
            "extract_text_failed",
            path=str(path),
            suffix=path.suffix,
            error=str(exc),
        )
        return ""


def is_rich_format(path: Path) -> bool:
    """Return True if *path* needs rich-format extraction."""
    return path.suffix.lower() in RICH_EXTENSIONS
