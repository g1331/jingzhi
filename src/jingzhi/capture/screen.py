from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from jingzhi.clock import SessionClock
from jingzhi.database import Database

logger = logging.getLogger(__name__)


def average_hash(image: Image.Image, size: int = 16) -> int:
    gray = image.convert("L").resize((size, size))
    pixels = gray.tobytes()
    mean = sum(pixels) / len(pixels)
    result = 0
    for index, value in enumerate(pixels):
        if value >= mean:
            result |= 1 << index
    return result


class ScreenCaptureWorker(threading.Thread):
    def __init__(
        self,
        *,
        database: Database,
        session_id: str,
        clock: SessionClock,
        output_dir: Path,
        stop_event: threading.Event,
        interval_s: float,
        hash_distance: int,
        on_frame: Callable[[int, Path], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name="screen-capture", daemon=True)
        self.database = database
        self.session_id = session_id
        self.clock = clock
        self.output_dir = output_dir
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.hash_distance = hash_distance
        self.on_frame = on_frame
        self.on_error = on_error

    def run(self) -> None:
        try:
            from mss import MSS

            self.output_dir.mkdir(parents=True, exist_ok=True)
            previous_hash: int | None = None
            with MSS() as capture:
                monitor = capture.primary_monitor
                while not self.stop_event.is_set():
                    shot = capture.grab(monitor)
                    image = Image.frombytes("RGB", shot.size, shot.rgb)
                    image_hash = average_hash(image)
                    distance = (
                        self.hash_distance + 1
                        if previous_hash is None
                        else (image_hash ^ previous_hash).bit_count()
                    )
                    if distance >= self.hash_distance:
                        ts_ms = self.clock.now_ms()
                        path = self.output_dir / f"{ts_ms:012d}.webp"
                        image.save(path, "WEBP", quality=76, method=4)
                        self.database.add_frame(
                            self.session_id, ts_ms, path, f"{image_hash:064x}", image.size
                        )
                        previous_hash = image_hash
                        if self.on_frame:
                            self.on_frame(ts_ms, path)
                    self.stop_event.wait(self.interval_s)
        except Exception as exc:  # a capture worker must report failure without killing the UI
            logger.exception("Screen capture failed")
            if self.on_error:
                self.on_error(f"Screen capture failed: {exc}")
