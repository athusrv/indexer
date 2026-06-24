"""Indexing subsystem — scanner, fingerprinter, chunker, extractor, pipeline."""

from context_db.indexer.chunker import Chunker
from context_db.indexer.extractor import RICH_EXTENSIONS, extract_text, is_rich_format
from context_db.indexer.fingerprint import Fingerprinter, fingerprint_file, hash_file
from context_db.indexer.pipeline import IndexingPipeline
from context_db.indexer.scanner import Scanner

__all__ = [
    "Scanner",
    "Fingerprinter",
    "fingerprint_file",
    "hash_file",
    "Chunker",
    "IndexingPipeline",
    "extract_text",
    "is_rich_format",
    "RICH_EXTENSIONS",
]
