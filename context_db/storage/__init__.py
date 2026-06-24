"""Storage layer — SQLite + FTS5."""

from context_db.storage.db import open_db
from context_db.storage.repository import Repository

__all__ = ["open_db", "Repository"]
