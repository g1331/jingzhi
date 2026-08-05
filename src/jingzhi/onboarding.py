from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ONBOARDING_STEPS = ("privacy", "provider", "whisper", "recording", "shortcuts")
ONBOARDING_VERSION = 1
DEFAULT_QUESTION_SHORTCUT = "Ctrl+Shift+Q"


@dataclass(frozen=True, slots=True)
class OnboardingState:
    step: str = ONBOARDING_STEPS[0]
    privacy_acknowledged: bool = False
    provider_completed: bool = False
    provider_skipped: bool = False
    whisper_completed: bool = False
    whisper_skipped: bool = False
    recording_confirmed: bool = False
    shortcuts_completed: bool = False
    question_shortcut: str = DEFAULT_QUESTION_SHORTCUT
    completed: bool = False

    def __post_init__(self) -> None:
        if self.step not in ONBOARDING_STEPS:
            object.__setattr__(self, "step", ONBOARDING_STEPS[0])
        if not isinstance(self.question_shortcut, str) or not self.question_shortcut.strip():
            object.__setattr__(self, "question_shortcut", DEFAULT_QUESTION_SHORTCUT)

    @property
    def step_index(self) -> int:
        return ONBOARDING_STEPS.index(self.step)

    @property
    def provider_ready(self) -> bool:
        return self.provider_completed or self.provider_skipped

    @property
    def whisper_ready(self) -> bool:
        return self.whisper_completed or self.whisper_skipped

    @property
    def ready_to_finish(self) -> bool:
        return (
            self.privacy_acknowledged
            and self.provider_ready
            and self.whisper_ready
            and self.recording_confirmed
            and self.shortcuts_completed
        )


class OnboardingSettingsStore:
    """Persists only onboarding progress; reset never touches application data."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "onboarding.json"

    def load(self) -> OnboardingState:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return OnboardingState()
        if (
            not isinstance(document, dict)
            or type(document.get("version")) is not int
            or document["version"] != ONBOARDING_VERSION
        ):
            return OnboardingState()
        values = {
            field: document.get(field, default)
            for field, default in asdict(OnboardingState()).items()
        }
        boolean_fields = (
            "privacy_acknowledged",
            "provider_completed",
            "provider_skipped",
            "whisper_completed",
            "whisper_skipped",
            "recording_confirmed",
            "shortcuts_completed",
            "completed",
        )
        for field in boolean_fields:
            values[field] = values[field] if isinstance(values[field], bool) else False
        values["step"] = str(values["step"])
        shortcut = values["question_shortcut"]
        values["question_shortcut"] = (
            shortcut
            if isinstance(shortcut, str) and shortcut.strip()
            else DEFAULT_QUESTION_SHORTCUT
        )
        state = OnboardingState(**values)
        if state.completed and not state.ready_to_finish:
            state = replace(state, completed=False)
        return state

    def save(self, state: OnboardingState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"version": ONBOARDING_VERSION, **asdict(state)}
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def reset(self) -> OnboardingState:
        current = self.load()
        state = OnboardingState(question_shortcut=current.question_shortcut)
        self.save(state)
        return state
