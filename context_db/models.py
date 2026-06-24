"""Shared Pydantic data models used across the entire pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Scanner output
# ---------------------------------------------------------------------------


class DiscoveredFile(BaseModel):
    """A file path found by the Scanner."""

    path: Path

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Fingerprinter output
# ---------------------------------------------------------------------------


class FileMetadata(BaseModel):
    """Fingerprint of a file — used for change detection."""

    path: Path
    hash: str  # hex-encoded SHA-256
    mtime: float  # Unix timestamp (float for sub-second precision)
    size: int  # bytes

    model_config = {"frozen": True}

    @field_validator("hash")
    @classmethod
    def _validate_hash(cls, v: str) -> str:
        if len(v) != 64:
            raise ValueError("hash must be a 64-char hex SHA-256 digest")
        return v


# ---------------------------------------------------------------------------
# Change detection result
# ---------------------------------------------------------------------------


class ChangeSet(BaseModel):
    """Delta produced by the Fingerprinter after comparing against the DB."""

    new_files: list[FileMetadata] = Field(default_factory=list)
    modified_files: list[FileMetadata] = Field(default_factory=list)
    deleted_paths: list[Path] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.modified_files or self.deleted_paths)

    @property
    def total_changes(self) -> int:
        return len(self.new_files) + len(self.modified_files) + len(self.deleted_paths)


# ---------------------------------------------------------------------------
# Chunker output
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A sub-document slice of a source file."""

    path: Path
    start_line: Annotated[int, Field(ge=1)]
    end_line: Annotated[int, Field(ge=1)]
    content: str

    model_config = {"frozen": True}

    @field_validator("end_line")
    @classmethod
    def _end_after_start(cls, v: int, info: object) -> int:
        # info.data may not have start_line if validation failed earlier
        data = getattr(info, "data", {})
        start = data.get("start_line")
        if start is not None and v < start:
            raise ValueError(f"end_line ({v}) must be >= start_line ({start})")
        return v


# ---------------------------------------------------------------------------
# Storage row models (hydrated from SQLite)
# ---------------------------------------------------------------------------


class StoredFile(BaseModel):
    """A file row as stored in the *files* table."""

    id: int
    path: Path
    hash: str
    mtime: float

    model_config = {"frozen": True}


class StoredChunk(BaseModel):
    """A chunk row as stored in the *chunks* table."""

    id: int
    file_id: int
    start_line: int
    end_line: int
    content: str

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A ranked result returned by the Search engine."""

    path: Path
    score: float
    start_line: int
    end_line: int
    preview: str  # first N chars of matching chunk

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Index stats
# ---------------------------------------------------------------------------


class IndexStats(BaseModel):
    """Summary statistics about the current index."""

    file_count: int
    chunk_count: int
    db_size_bytes: int


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


class ChunkFileInfo(BaseModel):
    """Chunk data joined with its source file path — used by hybrid retrieval."""

    chunk_id: int
    path: Path
    start_line: int
    end_line: int
    content: str

    model_config = {"frozen": True}


class HybridResult(BaseModel):
    """A ranked result from hybrid (lexical + semantic) search."""

    path: Path
    score: float  # weighted combination of normalised lexical + semantic scores
    match_type: Literal["lexical", "semantic", "hybrid"]
    start_line: int
    end_line: int
    preview: str

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Embedding storage metadata
# ---------------------------------------------------------------------------


class StoredEmbedding(BaseModel):
    """Metadata row from *chunk_embeddings* (vector not included)."""

    chunk_id: int
    model: str
    dimensions: int
    created_at: int  # Unix timestamp (seconds)

    model_config = {"frozen": True}
