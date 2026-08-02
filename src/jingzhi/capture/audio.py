from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from jingzhi.clock import SessionClock
from jingzhi.database import Database

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    id: int
    session_id: str
    source: str
    start_ms: int
    end_ms: int
    path: Path


class SoundDeviceMicrophoneRecorder:
    """Blocking microphone recorder backed by PortAudio.

    SoundCard's WASAPI implementation asserts that every microphone exposes
    WAVEFORMATEXTENSIBLE. Several valid Windows devices do not, so microphone
    input uses python-sounddevice while system loopback remains on SoundCard.
    """

    def __init__(self, requested_sample_rate: int, blocksize: int) -> None:
        self.requested_sample_rate = requested_sample_rate
        self.blocksize = blocksize
        self.sample_rate = requested_sample_rate
        self._stream = None

    def __enter__(self) -> Self:
        import sounddevice as sd

        device_index = sd.default.device[0]
        if device_index is None or int(device_index) < 0:
            raise RuntimeError("Windows 没有配置默认麦克风")
        device = sd.query_devices(device_index, "input")
        max_channels = int(device["max_input_channels"])
        if max_channels < 1:
            raise RuntimeError(f"设备“{device['name']}”没有输入通道")

        try:
            sd.check_input_settings(
                device=device_index,
                channels=1,
                dtype="float32",
                samplerate=self.requested_sample_rate,
            )
        except sd.PortAudioError:
            self.sample_rate = round(float(device["default_samplerate"]))
            logger.info(
                "Microphone does not support %s Hz; using device default %s Hz",
                self.requested_sample_rate,
                self.sample_rate,
            )

        self._stream = sd.InputStream(
            device=device_index,
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            latency="low",
        )
        self._stream.start()
        logger.info(
            "Microphone capture started through PortAudio: %s at %s Hz",
            device["name"],
            self.sample_rate,
        )
        return self

    def record(self, numframes: int) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("麦克风输入流尚未启动")
        data, overflowed = self._stream.read(numframes)
        if overflowed:
            logger.warning("Microphone input overflowed; some samples may be missing")
        return data.copy()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def _prepare_mono_audio(
    samples: np.ndarray, input_sample_rate: int, output_sample_rate: int
) -> np.ndarray:
    if samples.ndim == 1:
        samples = samples[:, None]
    mono = samples.mean(axis=1, dtype=np.float32)
    if input_sample_rate == output_sample_rate or not len(mono):
        return mono
    if input_sample_rate > output_sample_rate and input_sample_rate % output_sample_rate == 0:
        ratio = input_sample_rate // output_sample_rate
        usable = len(mono) - (len(mono) % ratio)
        return mono[:usable].reshape(-1, ratio).mean(axis=1, dtype=np.float32)

    output_length = round(len(mono) * output_sample_rate / input_sample_rate)
    source_positions = np.arange(output_length, dtype=np.float64) * (
        input_sample_rate / output_sample_rate
    )
    source_positions = np.minimum(source_positions, len(mono) - 1)
    return np.interp(source_positions, np.arange(len(mono)), mono).astype(np.float32)


def _write_flac(
    path: Path, samples: np.ndarray, input_sample_rate: int, output_sample_rate: int
) -> None:
    import soundfile as sf

    mono = _prepare_mono_audio(samples, input_sample_rate, output_sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(mono, -1.0, 1.0), output_sample_rate, format="FLAC", subtype="PCM_16")


class AudioCaptureWorker(threading.Thread):
    def __init__(
        self,
        *,
        database: Database,
        session_id: str,
        clock: SessionClock,
        source: str,
        output_dir: Path,
        stop_event: threading.Event,
        chunk_queue: queue.Queue[AudioChunk | None],
        sample_rate: int,
        storage_sample_rate: int,
        chunk_s: float,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name=f"audio-{source}", daemon=True)
        self.database = database
        self.session_id = session_id
        self.clock = clock
        self.source = source
        self.output_dir = output_dir
        self.stop_event = stop_event
        self.chunk_queue = chunk_queue
        self.sample_rate = sample_rate
        self.storage_sample_rate = storage_sample_rate
        self.chunk_s = chunk_s
        self.on_error = on_error

    def _open_recorder(self):
        import soundcard as sc

        if self.source == "microphone":
            return SoundDeviceMicrophoneRecorder(self.sample_rate, blocksize=8192)
        speaker = sc.default_speaker()
        loopback = sc.get_microphone(speaker.id, include_loopback=True)
        return loopback.recorder(samplerate=self.sample_rate, blocksize=8192)

    def run(self) -> None:
        try:
            with self._open_recorder() as recorder:
                active_sample_rate = int(getattr(recorder, "sample_rate", self.sample_rate))
                frames_per_chunk = int(active_sample_rate * self.chunk_s)
                block_frames = min(4096, frames_per_chunk)
                while not self.stop_event.is_set():
                    start_ms = self.clock.now_ms()
                    blocks: list[np.ndarray] = []
                    received = 0
                    while received < frames_per_chunk and not self.stop_event.is_set():
                        block = recorder.record(
                            numframes=min(block_frames, frames_per_chunk - received)
                        )
                        if block.size:
                            blocks.append(block)
                            received += len(block)
                    if not blocks:
                        continue
                    samples = np.concatenate(blocks)
                    end_ms = self.clock.now_ms()
                    path = self.output_dir / self.source / f"{start_ms:012d}-{end_ms:012d}.flac"
                    _write_flac(path, samples, active_sample_rate, self.storage_sample_rate)
                    chunk_id = self.database.add_audio_chunk(
                        self.session_id, self.source, start_ms, end_ms, path
                    )
                    chunk = AudioChunk(
                        id=chunk_id,
                        session_id=self.session_id,
                        source=self.source,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        path=path,
                    )
                    try:
                        self.chunk_queue.put_nowait(chunk)
                    except queue.Full:
                        # The chunk is already durable and remains "pending" for recovery.
                        # Capture must not deadlock merely because transcription is slower.
                        message = f"Transcription queue is full; {path.name} remains pending"
                        logger.warning(message)
                        if self.on_error:
                            self.on_error(message)
        except Exception as exc:
            logger.exception("%s audio capture failed", self.source)
            if self.on_error:
                detail = str(exc).strip() or type(exc).__name__
                source_name = "麦克风" if self.source == "microphone" else "系统声音"
                self.on_error(f"{source_name}采集不可用：{detail}")
