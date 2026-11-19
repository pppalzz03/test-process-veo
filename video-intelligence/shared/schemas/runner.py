"""Contract between the DeepStream runner subprocess and the worker (NDJSON on stdout).

One JSON object per line:
  {"type":"frame","frame_num":12,"pts_ms":400,"objects":[{...}]}
  {"type":"summary","frames":9000,"eos":true}
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ObjectRecord(BaseModel):
    track_id: int | None = None
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # left, top, width, height in source pixels


class FrameRecord(BaseModel):
    type: Literal["frame"] = "frame"
    frame_num: int
    pts_ms: int  # container PTS in milliseconds
    objects: list[ObjectRecord] = Field(default_factory=list)


class SummaryRecord(BaseModel):
    type: Literal["summary"] = "summary"
    frames: int
    eos: bool


RunnerRecord = Annotated[FrameRecord | SummaryRecord, Field(discriminator="type")]
_adapter: TypeAdapter[FrameRecord | SummaryRecord] = TypeAdapter(RunnerRecord)

# Exit codes of the runner process.
EXIT_OK = 0
EXIT_BAD_INPUT = 2  # corrupt / unreadable / unsupported video -> permanent
EXIT_PIPELINE = 3  # pipeline or GPU error -> transient
EXIT_ENVIRONMENT = 4  # pyds / engine / config missing -> retrying cannot help


def parse_runner_line(line: str) -> FrameRecord | SummaryRecord:
    """Parse one NDJSON line; raises ValueError on malformed input."""
    try:
        return _adapter.validate_python(json.loads(line))
    except Exception as exc:  # json or pydantic validation
        raise ValueError(f"invalid runner record: {line[:200]!r}") from exc
