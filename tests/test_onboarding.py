from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from jingzhi.onboarding import OnboardingSettingsStore, OnboardingState


def test_onboarding_progress_survives_restart_and_unknown_step_resets_safely(
    tmp_path: Path,
) -> None:
    store = OnboardingSettingsStore(tmp_path)
    state = replace(
        OnboardingState(),
        step="whisper",
        privacy_acknowledged=True,
        provider_skipped=True,
    )
    store.save(state)

    assert store.load() == state
    assert store.load().step_index == 2
    assert store.load().provider_ready

    store.path.write_text(
        json.dumps({"version": 1, "step": "not-a-step", "completed": "yes"}),
        encoding="utf-8",
    )
    recovered = store.load()
    assert recovered.step == "privacy"
    assert recovered.completed is False
    assert recovered.privacy_acknowledged is False

    store.path.write_text(
        json.dumps({"version": 1, "completed": True}),
        encoding="utf-8",
    )
    inconsistent = store.load()
    assert inconsistent.completed is False

    store.path.write_text(
        json.dumps({"version": 1, "question_shortcut": None}),
        encoding="utf-8",
    )
    assert store.load().question_shortcut == "Ctrl+Shift+Q"

    store.path.write_bytes(b'{"version": 1, "step": \xff}')
    assert store.load() == OnboardingState()

    store.path.write_text(
        json.dumps({"version": True, "completed": True}),
        encoding="utf-8",
    )
    assert store.load() == OnboardingState()


def test_reset_only_restarts_onboarding_and_preserves_application_files(tmp_path: Path) -> None:
    store = OnboardingSettingsStore(tmp_path)
    application_file = tmp_path / "sessions" / "keep.txt"
    application_file.parent.mkdir()
    application_file.write_text("session data", encoding="utf-8")
    store.save(
        replace(
            OnboardingState(),
            completed=True,
            step="shortcuts",
            question_shortcut="Alt+F8",
        )
    )

    reset = store.reset()

    assert reset.completed is False
    assert reset.question_shortcut == "Alt+F8"
    assert store.load().completed is False
    assert store.load().question_shortcut == "Alt+F8"
    assert application_file.read_text(encoding="utf-8") == "session data"


def test_state_readiness_distinguishes_skip_from_incomplete_steps() -> None:
    incomplete = replace(OnboardingState(), privacy_acknowledged=True)
    ready = replace(
        incomplete,
        provider_skipped=True,
        whisper_completed=True,
        recording_confirmed=True,
        shortcuts_completed=True,
    )

    assert incomplete.ready_to_finish is False
    assert ready.provider_ready
    assert ready.whisper_ready
    assert ready.ready_to_finish
