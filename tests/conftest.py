"""Shared pytest fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Generator

import pytest
import structlog

from context_db.storage.db import open_db
from context_db.storage.repository import Repository


@pytest.fixture(autouse=True)
def reset_structlog_after_test() -> Generator[None, None, None]:
    """Reset structlog to WARNING level after each test.

    Prevents the ``--verbose`` CLI test from permanently lowering the log level
    to DEBUG and polluting subsequent tests' captured output.
    """
    yield
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(30)  # WARNING
    )


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Generator[tuple[Repository, sqlite3.Connection, Path], None, None]:
    """Return a fresh (Repository, connection, db_path) triple."""
    db_path = tmp_path / "test.db"
    conn = open_db(db_path)
    repo = Repository(conn)
    yield repo, conn, db_path
    conn.close()


@pytest.fixture()
def sample_tree(tmp_path: Path) -> Path:
    """Build a small directory tree for integration tests.

    Structure
    ---------
    root/
        src/
            auth.py        (contains 'jwt', 'token')
            utils.py       (contains 'helper', 'format')
        README.md          (contains 'context-db')
        node_modules/
            ignored.js     (should NOT be indexed)
        __pycache__/
            cached.pyc     (should NOT be indexed — binary ext)
    """
    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    (root / "node_modules").mkdir()
    (root / "__pycache__").mkdir()

    (root / "src" / "auth.py").write_text(
        "# Authentication module\n"
        "import jwt\n"
        "\n"
        "def verify_token(token: str) -> bool:\n"
        "    payload = jwt.decode(token, 'secret', algorithms=['HS256'])\n"
        "    return payload is not None\n",
        encoding="utf-8",
    )
    (root / "src" / "utils.py").write_text(
        "# Utility helpers\n"
        "\n"
        "def format_date(ts: float) -> str:\n"
        "    return str(int(ts))\n"
        "\n"
        "def helper(x: int) -> int:\n"
        "    return x * 2\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# context-db\n\nA filesystem indexer for AI agents.\n",
        encoding="utf-8",
    )
    (root / "node_modules" / "ignored.js").write_text(
        "console.log('should not be indexed');",
        encoding="utf-8",
    )
    (root / "__pycache__" / "cached.pyc").write_bytes(b"\x00binary\x00")
    return root
