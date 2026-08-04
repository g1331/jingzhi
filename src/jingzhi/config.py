from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from jingzhi.provider_settings import ProviderSettingsStore
from jingzhi.transcript_correction import CORRECTION_WINDOW_SECONDS


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    screen_interval_s: float = 1.0
    screen_hash_distance: int = 10
    audio_capture_rate: int = 48_000
    audio_storage_rate: int = 16_000
    audio_chunk_s: float = 8.0
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    llm_model: str = "gpt-5.5"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_api_mode: str = "responses"
    capture_microphone: bool = True
    capture_system_audio: bool = True
    transcript_correction_enabled: bool = False
    transcript_correction_window_seconds: int = CORRECTION_WINDOW_SECONDS[1]
    transcript_correction_model: str = "gpt-5.5"

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("STUDY_DATA_DIR", "data")).resolve()
        saved = ProviderSettingsStore(data_dir).load()
        return cls(
            data_dir=data_dir,
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            llm_model=os.getenv("OPENAI_MODEL", saved.model),
            llm_base_url=os.getenv("OPENAI_BASE_URL", saved.base_url),
            llm_api_key=os.getenv("OPENAI_API_KEY", saved.api_key),
            llm_api_mode=os.getenv("OPENAI_API_MODE", saved.api_mode),
            transcript_correction_model=os.getenv(
                "TRANSCRIPT_CORRECTION_MODEL", saved.correction_model
            ),
        )
