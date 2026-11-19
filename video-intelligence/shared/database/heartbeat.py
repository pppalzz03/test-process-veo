"""Service liveness via a heartbeat row (used by container healthchecks for non-HTTP services)."""

from __future__ import annotations

import sys

import psycopg

from shared.database.pool import DictConnection


def beat(conn: DictConnection, service: str) -> None:
    with conn.transaction():
        conn.execute(
            "INSERT INTO service_heartbeats (service, beat_at) VALUES (%s, now())"
            " ON CONFLICT (service) DO UPDATE SET beat_at = now()",
            (service,),
        )


def seconds_since_beat(database_url: str, service: str) -> float | None:
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            "SELECT extract(epoch FROM now() - beat_at) FROM service_heartbeats WHERE service=%s",
            (service,),
        ).fetchone()
    return float(row[0]) if row else None


if __name__ == "__main__":
    # usage: python -m shared.database.heartbeat <service> <max_age_seconds>
    from shared.config.settings import get_settings

    service_name, max_age = sys.argv[1], float(sys.argv[2])
    age = seconds_since_beat(get_settings().database_url, service_name)
    sys.exit(0 if age is not None and age <= max_age else 1)
