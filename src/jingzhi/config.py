from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from jingzhi.model_roles import RoleName, default_roles
from jingzhi.provider_settings import (
    ProviderSettingsStore,
    SavedProviderSettings,
    default_saved_settings,
)
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
    provider_settings: SavedProviderSettings = field(default_factory=default_saved_settings)
    capture_microphone: bool = True
    capture_system_audio: bool = True
    transcript_correction_enabled: bool = False
    transcript_correction_window_seconds: int = CORRECTION_WINDOW_SECONDS[1]

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("STUDY_DATA_DIR", "data")).resolve()
        saved = ProviderSettingsStore(data_dir).load()
        connections = list(saved.connections or default_saved_settings().connections)
        primary = connections[0]
        primary = replace(
            primary,
            base_url=os.getenv("OPENAI_BASE_URL", primary.base_url),
            api_key=os.getenv("OPENAI_API_KEY", primary.api_key),
            api_mode=os.getenv("OPENAI_API_MODE", primary.api_mode),
        )
        connections[0] = primary
        configured_roles = {role.name: role for role in saved.roles}
        defaults = {role.name: replace(role, connection_id=primary.id) for role in default_roles()}
        defaults.update(configured_roles)
        main_model = os.getenv("OPENAI_MODEL")
        correction_model = os.getenv("TRANSCRIPT_CORRECTION_MODEL")
        if main_model:
            for name in (RoleName.UTILITY, RoleName.INSTANT_ANSWER, RoleName.DEEP_ANALYSIS):
                defaults[name] = replace(defaults[name], model=main_model)
        if correction_model:
            defaults[RoleName.TRANSCRIPT_CORRECTION] = replace(
                defaults[RoleName.TRANSCRIPT_CORRECTION], model=correction_model
            )
        provider_settings = SavedProviderSettings(
            tuple(connections),
            tuple(defaults[role.name] for role in default_roles()),
        )
        return cls(
            data_dir=data_dir,
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            provider_settings=provider_settings,
        )
