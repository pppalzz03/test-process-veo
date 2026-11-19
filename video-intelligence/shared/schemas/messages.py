"""Queue message contracts. Messages carry IDs only - never video bytes."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel

SCHEMA_VERSION = 1


class JobMessage(BaseModel):
    schema_version: int = SCHEMA_VERSION
    task_type: Literal["video.infer"] = "video.infer"
    job_id: UUID
    attempt: int = 0  # informational; the authoritative counter is jobs.attempts


class DeadLetterMessage(BaseModel):
    schema_version: int = SCHEMA_VERSION
    task_type: Literal["video.dead_letter"] = "video.dead_letter"
    job_id: UUID
    reason_code: str
    reason: str
    attempts: int
