from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEYRING_SERVICE = "JINGZHI"
LEGACY_KEYRING_SERVICE = "Study Companion"
KEYRING_USERNAME = "openai-compatible-api-key"


@dataclass(frozen=True, slots=True)
class SavedProviderSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-5.5"
    correction_model: str = ""
    api_mode: str = "responses"


class ProviderSettingsStore:
    """Persists provider metadata and keeps the API key in the OS keyring."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "provider.json"

    def load(self) -> SavedProviderSettings:
        public: dict[str, Any] = {}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                public = loaded

        import keyring
        from keyring.errors import KeyringError

        try:
            api_key = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or ""
            if not api_key:
                api_key = keyring.get_password(LEGACY_KEYRING_SERVICE, KEYRING_USERNAME) or ""
        except KeyringError:
            api_key = ""

        api_mode = str(public.get("api_mode", "responses"))
        if api_mode not in {"responses", "chat_completions"}:
            api_mode = "responses"
        model = str(public.get("model", "gpt-5.5"))
        correction_model = str(public.get("correction_model", "")).strip() or model
        return SavedProviderSettings(
            base_url=str(public.get("base_url", "")),
            api_key=api_key,
            model=model,
            correction_model=correction_model,
            api_mode=api_mode,
        )

    def save(self, settings: SavedProviderSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        public = {
            "version": 1,
            "base_url": settings.base_url.strip(),
            "model": settings.model.strip(),
            "correction_model": settings.correction_model.strip() or settings.model.strip(),
            "api_mode": settings.api_mode,
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)

        import keyring
        from keyring.errors import KeyringError

        try:
            if settings.api_key:
                keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, settings.api_key)
            elif keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) is not None:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except KeyringError as exc:
            raise RuntimeError("无法写入 Windows 凭据管理器，API Key 未保存") from exc
