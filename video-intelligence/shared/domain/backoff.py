"""Exponential backoff with jitter."""

from __future__ import annotations

import random


def compute_backoff_seconds(
    attempt: int,
    *,
    base_seconds: float,
    max_seconds: float,
    jitter_ratio: float = 0.2,
    rng: random.Random | None = None,
) -> float:
    """Delay before retry number `attempt` (1 = first retry).

    delay = min(max, base * 2**(attempt-1)), then +/- jitter_ratio so many failed
    jobs do not retry in lockstep.
    """
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    raw = min(max_seconds, base_seconds * (2 ** (attempt - 1)))
    rand = (rng or random).random()
    jitter = raw * jitter_ratio * (2 * rand - 1)
    return max(0.0, min(max_seconds, raw + jitter))
