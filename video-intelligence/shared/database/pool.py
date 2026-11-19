from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# Rows are dicts (row_factory=dict_row), so annotate connections accordingly.
DictConnection = Connection[dict[str, Any]]
DictPool = ConnectionPool[DictConnection]


def create_pool(database_url: str, *, min_size: int = 1, max_size: int = 8) -> DictPool:
    """Rows come back as dicts. The pool commits on clean exit of `with pool.connection()`."""
    pool: DictPool = ConnectionPool(  # type: ignore[assignment]  # dict_row via kwargs
        database_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    pool.open(wait=True, timeout=30)
    return pool
