from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path

from jingzhi.storage import storage_writer


class MaterialGenerationMode(StrEnum):
    ALWAYS = "always"
    ASK = "ask"
    MANUAL = "manual"


class MaterialGenerationSettingsStore:
    """Stores the user's decision for when finished sessions may generate material."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "material.json"

    def load(self) -> MaterialGenerationMode | None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(document, dict):
            return None
        try:
            return MaterialGenerationMode(str(document["generation_mode"]))
        except (KeyError, TypeError, ValueError):
            return None

    @storage_writer("保存会话材料生成策略")
    def save(self, mode: MaterialGenerationMode) -> None:
        if not isinstance(mode, MaterialGenerationMode):
            raise TypeError("Unsupported material generation mode")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"version": 1, "generation_mode": mode.value},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
