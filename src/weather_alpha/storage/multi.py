from __future__ import annotations

import logging
from typing import Protocol

from ..domain import EventEnvelope

logger = logging.getLogger(__name__)


class CriticalSinkError(RuntimeError):
    pass


class EventSink(Protocol):
    async def append(self, event: EventEnvelope) -> None: ...

    async def close(self) -> None: ...


class MultiSink:
    def __init__(
        self,
        required_sinks: list[EventSink],
        best_effort_sinks: list[EventSink] | None = None,
    ) -> None:
        if not required_sinks:
            raise ValueError("At least one required event sink must be configured")
        self.required_sinks = required_sinks
        self.best_effort_sinks = best_effort_sinks or []

    async def append(self, event: EventEnvelope) -> None:
        for sink in self.required_sinks:
            try:
                await sink.append(event)
            except Exception as exc:
                raise CriticalSinkError(
                    f"Required event sink failed: {type(sink).__name__}"
                ) from exc
        for sink in self.best_effort_sinks:
            try:
                await sink.append(event)
            except Exception:
                logger.exception("Best-effort event sink failed: %s", type(sink).__name__)

    async def close(self) -> None:
        for sink in self.best_effort_sinks:
            try:
                await sink.close()
            except Exception:
                logger.exception("Failed to close best-effort sink: %s", type(sink).__name__)
        for sink in self.required_sinks:
            try:
                await sink.close()
            except Exception as exc:
                raise CriticalSinkError(
                    f"Failed to close required event sink: {type(sink).__name__}"
                ) from exc
