"""Minimal forward-only SQL migration runner (no Alembic: one schema, plain SQL files)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import psycopg

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(
    os.environ.get("MIGRATIONS_DIR", str(Path(__file__).resolve().parents[2] / "migrations"))
)
_LOCK_KEY = 7_202_611  # arbitrary advisory-lock id so concurrent runners serialize


class UnsupportedDatabaseError(RuntimeError):
    """The target database cannot be used safely by this application."""


def _require_utf8(conn: psycopg.Connection) -> None:
    """Text columns come back as bytes from a SQL_ASCII database, which silently breaks lookups
    and JSON responses. Fail early with an actionable message instead."""
    row = conn.execute("SHOW server_encoding").fetchone()
    assert row is not None
    encoding = row[0].decode() if isinstance(row[0], bytes) else row[0]
    if encoding.upper() != "UTF8":
        raise UnsupportedDatabaseError(
            f"database encoding is {encoding}; create it with ENCODING 'UTF8' "
            "(e.g. initdb -E UTF8 / CREATE DATABASE ... ENCODING 'UTF8' TEMPLATE template0)"
        )


def apply_migrations(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending *.sql files in name order; returns the names applied."""
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as conn:
        _require_utf8(conn)
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
            for path in sorted(migrations_dir.glob("*.sql")):
                if path.name in done:
                    continue
                with conn.transaction():
                    conn.execute(path.read_text())
                    conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
                logger.info("migration applied", extra={"migration": path.name})
                applied.append(path.name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    return applied


if __name__ == "__main__":
    from shared.config.settings import get_settings
    from shared.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json, service="migrate")
    names = apply_migrations(settings.database_url)
    logger.info("migrations complete", extra={"applied": names})
    sys.exit(0)
