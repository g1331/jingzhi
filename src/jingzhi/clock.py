from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class SessionClock:
    """Maps monotonic time to one session-relative timeline.

    Capture workers must not align data with wall-clock timestamps because NTP and manual
    clock changes can move wall time backwards. Persisting both values keeps exports readable.
    """

    started_monotonic_ns: int
    started_at_utc: str

    @classmethod
    def start(cls) -> SessionClock:
        return cls(
            started_monotonic_ns=time.monotonic_ns(),
            started_at_utc=datetime.now(UTC).isoformat(),
        )

    def now_ms(self) -> int:
        return (time.monotonic_ns() - self.started_monotonic_ns) // 1_000_000
