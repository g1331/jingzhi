from __future__ import annotations

import logging
import queue
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from jingzhi.capture.devices import AudioDevice, DeviceCatalog
from jingzhi.clock import SessionClock
from jingzhi.database import Database

logger = logging.getLogger(__name__)
_SOUNDCARD_WARNING_LOCK = threading.Lock()
EMPTY_STREAM_TIMEOUT_MS = 5_000
OVERFLOW_TIMEOUT_MS = 5_000


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

    def __init__(
        self, requested_sample_rate: int, blocksize: int, device_index: int | None = None
    ) -> None:
        self.requested_sample_rate = requested_sample_rate
        self.blocksize = blocksize
        self.device_index = device_index
        self.sample_rate = requested_sample_rate
        self._stream = None
        self.last_overflowed = False

    def __enter__(self) -> Self:
        import sounddevice as sd

        device_index = self.device_index
        if device_index is None:
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
        self.last_overflowed = bool(overflowed)
        if overflowed:
            logger.warning("Microphone input overflowed; some samples may be missing")
        return data.copy()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SelectedMicrophoneRecorder:
    """Uses the exact MMDevice endpoint, with PortAudio fallback only when unambiguous."""

    def __init__(
        self,
        endpoint_id: Any,
        portaudio_index: int | None,
        requested_sample_rate: int,
        blocksize: int,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.portaudio_index = portaudio_index
        self.requested_sample_rate = requested_sample_rate
        self.blocksize = blocksize
        self.sample_rate = requested_sample_rate
        self._context = None
        self._recorder = None
        self.last_overflowed = False

    def __enter__(self) -> Self:
        import soundcard as sc

        microphone = sc.get_microphone(self.endpoint_id)
        self._context = microphone.recorder(
            samplerate=self.requested_sample_rate, blocksize=self.blocksize
        )
        try:
            self._recorder = self._context.__enter__()
        except Exception:
            if self.portaudio_index is None:
                raise
            logger.info("SoundCard could not open selected endpoint; using its PortAudio match")
            self._context = SoundDeviceMicrophoneRecorder(
                self.requested_sample_rate,
                self.blocksize,
                device_index=self.portaudio_index,
            )
            self._recorder = self._context.__enter__()
            self.sample_rate = self._recorder.sample_rate
        return self

    def record(self, numframes: int) -> np.ndarray:
        if self._recorder is None:
            raise RuntimeError("麦克风输入流尚未启动")
        data = self._recorder.record(numframes=numframes)
        self.last_overflowed = bool(
            getattr(self._recorder, "last_overflowed", False)
            or getattr(self._context, "last_overflowed", False)
        )
        return data

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        if self._context is not None:
            self._context.__exit__(exc_type, exc_value, traceback)
        self._context = None
        self._recorder = None


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


class QuestionVoiceRecorder:
    """Records one press-to-talk question clip from the default microphone."""

    def __init__(self, requested_sample_rate: int, storage_sample_rate: int) -> None:
        self.requested_sample_rate = requested_sample_rate
        self.storage_sample_rate = storage_sample_rate
        self._path: Path | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._error: Exception | None = None

    def start(self, path: Path) -> None:
        if self._thread is not None:
            raise RuntimeError("A question voice recording is already active")
        self._path = path
        self._stop_event.clear()
        self._ready_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._record, name="question-voice-capture", daemon=True
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=5):
            self._stop_event.set()
            raise RuntimeError("Microphone did not become ready")
        if self._error is not None:
            error = self._error
            self._thread = None
            raise error

    def _record(self) -> None:
        blocks: list[np.ndarray] = []
        try:
            with SoundDeviceMicrophoneRecorder(
                self.requested_sample_rate, blocksize=2048
            ) as recorder:
                active_sample_rate = int(recorder.sample_rate)
                self._ready_event.set()
                while not self._stop_event.is_set():
                    block = recorder.record(2048)
                    if block.size:
                        blocks.append(block)
            if not blocks:
                raise RuntimeError("Question voice recording is too short")
            assert self._path is not None
            _write_flac(
                self._path,
                np.concatenate(blocks),
                active_sample_rate,
                self.storage_sample_rate,
            )
        except Exception as exc:  # noqa: BLE001 - transferred to the UI thread
            self._error = exc
        finally:
            self._ready_event.set()

    def stop(self) -> Path:
        thread = self._thread
        path = self._path
        if thread is None or path is None:
            raise RuntimeError("No question voice recording is active")
        self._stop_event.set()
        thread.join(timeout=10)
        self._thread = None
        self._path = None
        if thread.is_alive():
            raise RuntimeError("Question voice recording did not stop")
        if self._error is not None:
            raise self._error
        if not path.exists():
            raise RuntimeError("Question voice recording was not saved")
        return path

    def cancel(self) -> None:
        thread = self._thread
        path = self._path
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=10)
        self._thread = None
        self._path = None
        if thread.is_alive():
            raise RuntimeError("Question voice recording did not stop")
        if path is not None:
            path.unlink(missing_ok=True)


class AudioCaptureWorker(threading.Thread):
    def __init__(
        self,
        *,
        database: Database,
        session_id: str,
        clock: SessionClock,
        source: str,
        device: AudioDevice | None,
        device_catalog: DeviceCatalog,
        output_dir: Path,
        stop_event: threading.Event,
        chunk_queue: queue.Queue[AudioChunk | None],
        sample_rate: int,
        storage_sample_rate: int,
        chunk_s: float,
        pause_event: threading.Event | None = None,
        on_error: Callable[[str], None] | None = None,
        on_failure: Callable[[str, str, int, int, str], None] | None = None,
    ) -> None:
        super().__init__(name=f"audio-{source}", daemon=True)
        self.database = database
        self.session_id = session_id
        self.clock = clock
        self.source = source
        self.device = device
        self.device_catalog = device_catalog
        self.output_dir = output_dir
        self.stop_event = stop_event
        self.pause_event = pause_event if pause_event is not None else threading.Event()
        self.chunk_queue = chunk_queue
        self.sample_rate = sample_rate
        self.storage_sample_rate = storage_sample_rate
        self.chunk_s = chunk_s
        self.on_error = on_error
        self.on_failure = on_failure
        self._last_success_ms = 0

    def _open_recorder(self):
        import soundcard as sc

        if self.device is None:
            raise RuntimeError(f"{self.source} audio source is unavailable")
        endpoint_id, portaudio_index = self.device_catalog.audio_locator(self.device.id)
        if self.source == "microphone":
            return SelectedMicrophoneRecorder(
                endpoint_id,
                portaudio_index,
                self.sample_rate,
                blocksize=8192,
            )
        speaker = sc.get_speaker(endpoint_id)
        loopback = sc.get_microphone(speaker.id, include_loopback=True)
        return loopback.recorder(samplerate=self.sample_rate, blocksize=8192)

    def _report_failure(self, kind: str, start_ms: int, end_ms: int, message: str) -> None:
        if self.on_failure:
            self.on_failure(self.source, kind, start_ms, end_ms, message)
        elif self.on_error:
            source_name = "麦克风" if self.source == "microphone" else "系统声音"
            self.on_error(f"{source_name}采集不可用：{message}")

    @staticmethod
    def _record_block(recorder: Any, numframes: int) -> tuple[np.ndarray, bool]:
        # warnings filters and showwarning are process-global; serialize the short
        # backend call so concurrent audio workers cannot steal each other's warning.
        with _SOUNDCARD_WARNING_LOCK, warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            block = recorder.record(numframes=numframes)
        overflowed = bool(getattr(recorder, "last_overflowed", False)) or any(
            warning.category.__name__ == "SoundcardRuntimeWarning" for warning in caught_warnings
        )
        return block, overflowed

    def _capture_open_recorder(self, recorder: Any) -> bool:
        active_sample_rate = int(getattr(recorder, "sample_rate", self.sample_rate))
        frames_per_chunk = int(active_sample_rate * self.chunk_s)
        block_frames = min(4096, frames_per_chunk)
        empty_started_ms: int | None = None
        overflow_started_ms: int | None = None
        while not self.stop_event.is_set() and not self.pause_event.is_set():
            start_ms = self.clock.now_ms()
            blocks: list[np.ndarray] = []
            received = 0
            while (
                received < frames_per_chunk
                and not self.stop_event.is_set()
                and not self.pause_event.is_set()
            ):
                block, overflowed = self._record_block(
                    recorder, min(block_frames, frames_per_chunk - received)
                )
                now_ms = self.clock.now_ms()
                self._last_success_ms = now_ms
                if overflowed:
                    if overflow_started_ms is None:
                        overflow_started_ms = now_ms
                    elif now_ms - overflow_started_ms >= OVERFLOW_TIMEOUT_MS:
                        self._report_failure(
                            "overflow",
                            overflow_started_ms,
                            now_ms,
                            "音频来源持续溢出，可能存在数据缺失",
                        )
                        return False
                else:
                    overflow_started_ms = None
                if block.size:
                    blocks.append(block)
                    received += len(block)
                else:
                    break
            paused = self.pause_event.is_set()
            if paused and not blocks:
                return True
            if not blocks:
                now_ms = self.clock.now_ms()
                if empty_started_ms is None:
                    empty_started_ms = now_ms
                elif now_ms - empty_started_ms >= EMPTY_STREAM_TIMEOUT_MS:
                    self._report_failure(
                        "stream_stopped",
                        empty_started_ms,
                        now_ms,
                        "音频来源持续无数据",
                    )
                    return False
                continue
            empty_started_ms = None
            if paused:
                overflow_started_ms = None
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
        return True

    def run(self) -> None:
        self._last_success_ms = self.clock.now_ms()
        recorder_opened = False
        try:
            while not self.stop_event.is_set():
                while self.pause_event.is_set() and not self.stop_event.is_set():
                    self._last_success_ms = self.clock.now_ms()
                    self.stop_event.wait(0.1)
                if self.stop_event.is_set():
                    break
                recorder_opened = False
                with self._open_recorder() as recorder:
                    recorder_opened = True
                    if not self._capture_open_recorder(recorder):
                        return
        except Exception as exc:
            logger.exception("%s audio capture failed", self.source)
            if self.stop_event.is_set():
                return
            detail = str(exc).strip() or type(exc).__name__
            kind = "failure" if recorder_opened else "device_unavailable"
            self._report_failure(kind, self._last_success_ms, self.clock.now_ms(), detail)
