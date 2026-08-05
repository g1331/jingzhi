import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QDialog

from jingzhi.capture.devices import RecordingSelection
from jingzhi.config import Settings
from jingzhi.model_roles import RoleName
from jingzhi.onboarding import OnboardingSettingsStore, OnboardingState
from jingzhi.provider_settings import default_saved_settings
from jingzhi.ui import MainWindow, OnboardingDialog
from jingzhi.whisper_settings import PROFILE_PRESETS, WhisperProfile


class FakeManager:
    def __init__(self, *, provider_settings=None, whisper_settings=None) -> None:
        self.provider_settings = provider_settings or default_saved_settings()
        self.whisper_settings = (
            whisper_settings or PROFILE_PRESETS[WhisperProfile.BALANCED].settings
        )
        self.device_catalog = object()
        self.saved_provider = False
        self.saved_whisper = False
        self.provider_tests = 0
        self.benchmark_calls = 0

    def configure_provider(self, settings) -> None:
        self.provider_settings = settings

    def save_provider(self) -> None:
        self.saved_provider = True

    def test_provider(self) -> str:
        self.provider_tests += 1
        return "连接正常"

    def configure_whisper(self, settings) -> None:
        self.whisper_settings = settings

    def save_whisper(self) -> None:
        self.saved_whisper = True

    def benchmark_whisper(self, _sample):
        self.benchmark_calls += 1
        self.whisper_settings = replace(self.whisper_settings, first_run_completed=True)
        return SimpleNamespace(elapsed_seconds=2.5, realtime_factor=1.2)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_onboarding_reopens_at_last_step_without_marking_complete(tmp_path: Path) -> None:
    _application()
    manager = FakeManager()
    dialog = OnboardingDialog(manager, Settings(data_dir=tmp_path))
    dialog.privacy_check.setChecked(True)
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 1
    dialog.reject()

    reopened = OnboardingDialog(manager, Settings(data_dir=tmp_path))
    assert reopened.pages.currentIndex() == 1
    assert reopened.state.privacy_acknowledged is True
    assert reopened.state.completed is False
    reopened.reject()


def test_onboarding_skip_and_finish_persists_choices(tmp_path: Path) -> None:
    _application()
    manager = FakeManager()
    dialog = OnboardingDialog(manager, Settings(data_dir=tmp_path))
    dialog.privacy_check.setChecked(True)
    dialog.next_button.click()
    dialog.skip_button.click()
    assert dialog.pages.currentIndex() == 2
    dialog.skip_button.click()
    assert dialog.pages.currentIndex() == 3
    dialog._save_state(recording_confirmed=True)
    dialog._set_page(4)
    dialog._confirm_shortcut()
    assert dialog.next_button.isEnabled()
    dialog.next_button.click()
    assert OnboardingSettingsStore(tmp_path).load().completed is False
    dialog.mark_completed()

    state = OnboardingSettingsStore(tmp_path).load()
    assert state.completed is True
    assert state.provider_skipped is True
    assert state.whisper_skipped is True
    assert manager.saved_whisper is True


def test_provider_form_edits_utility_connection_when_it_is_not_first(tmp_path: Path) -> None:
    _application()
    provider = default_saved_settings()
    utility_connection = replace(
        provider.connections[0],
        id="utility-connection",
        base_url="https://utility.example",
    )
    roles = tuple(
        replace(role, connection_id=utility_connection.id)
        if role.name == RoleName.UTILITY
        else role
        for role in provider.roles
    )
    provider = replace(
        provider, connections=(utility_connection, *provider.connections[1:]), roles=roles
    )
    dialog = OnboardingDialog(
        FakeManager(provider_settings=provider),
        Settings(data_dir=tmp_path),
    )
    assert dialog.onboarding_provider_url.text() == "https://utility.example"
    dialog.onboarding_provider_url.setText("https://changed.example")
    dialog.onboarding_provider_api_mode.setCurrentIndex(
        dialog.onboarding_provider_api_mode.findData("chat_completions")
    )
    settings = dialog._provider_settings_from_form()
    utility_role = next(role for role in settings.roles if role.name == RoleName.UTILITY)
    target = next(
        connection
        for connection in settings.connections
        if connection.id == utility_role.connection_id
    )
    assert target.base_url == "https://changed.example"
    assert target.api_mode == "chat_completions"
    dialog.reject()


def test_recording_capsule_preview_uses_shared_interactive_controls(tmp_path: Path) -> None:
    _application()
    dialog = OnboardingDialog(FakeManager(), Settings(data_dir=tmp_path))
    assert dialog.capsule_preview.objectName() == "recordingCapsulePreview"
    assert dialog.capsule_preview_question is dialog.capsule_preview.capsule_ask_button
    assert dialog.capsule_preview_start is dialog.capsule_preview.start_button
    assert dialog.capsule_preview.pause_button.isEnabled() is False
    dialog.reject()


def test_onboarding_uses_real_recording_confirmation_result(tmp_path: Path, monkeypatch) -> None:
    _application()
    selection = RecordingSelection(("display:1",), "system:1", "mic:1", 90)
    received: dict[str, object] = {}

    class FakeRecordingConfirmationDialog:
        def __init__(self, *_args, **kwargs) -> None:
            received.update(kwargs)

        def exec(self) -> int:
            return QDialog.DialogCode.Accepted

        def recording_selection(self) -> RecordingSelection:
            return selection

    monkeypatch.setattr("jingzhi.ui.RecordingConfirmationDialog", FakeRecordingConfirmationDialog)
    dialog = OnboardingDialog(FakeManager(), Settings(data_dir=tmp_path))
    dialog.capsule_preview.system_audio_check.setChecked(False)
    dialog._open_recording_confirmation()
    assert received["default_system_audio_enabled"] is False
    assert dialog.state.recording_confirmed is True
    assert dialog.recording_selection == selection
    dialog.reject()


def test_recording_failure_is_reported_as_recording_failure(tmp_path: Path, monkeypatch) -> None:
    _application()

    def fail_confirmation(*_args, **_kwargs):
        raise RuntimeError("没有可用显示器")

    monkeypatch.setattr("jingzhi.ui._confirm_recording_selection", fail_confirmation)
    dialog = OnboardingDialog(FakeManager(), Settings(data_dir=tmp_path))
    dialog._open_recording_confirmation()
    assert "录制确认失败" in dialog.recording_status.text()
    assert "模型连接失败" not in dialog.provider_status.text()
    dialog.reject()


def test_existing_configuration_is_prefilled_and_reset_keeps_user_data(tmp_path: Path) -> None:
    _application()
    provider = default_saved_settings()
    connection = replace(provider.connections[0], base_url="https://example.test", api_key="secret")
    provider = replace(provider, connections=(connection, *provider.connections[1:]))
    whisper = replace(
        PROFILE_PRESETS[WhisperProfile.ACCURATE].settings,
        first_run_completed=True,
    )
    (tmp_path / "recording.json").write_text(
        json.dumps(
            {
                "display_ids": ["display:2"],
                "system_audio_id": "system:2",
                "microphone_id": "mic:2",
                "system_audio_enabled": True,
                "microphone_enabled": False,
                "estimated_duration_minutes": 30,
            }
        ),
        encoding="utf-8",
    )
    sentinel = tmp_path / "sessions.sqlite3"
    sentinel.write_text("keep", encoding="utf-8")
    dialog = OnboardingDialog(
        FakeManager(provider_settings=provider, whisper_settings=whisper),
        Settings(data_dir=tmp_path),
    )
    assert dialog.state.provider_completed is True
    assert dialog.state.whisper_completed is True
    assert dialog.state.recording_confirmed is True
    assert "已保存录制来源配置" in dialog.recording_status.text()
    assert dialog.onboarding_provider_url.text() == "https://example.test"
    assert dialog.onboarding_whisper_profile.currentData() == WhisperProfile.ACCURATE.value
    dialog.state_store.reset()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert dialog.state_store.load().completed is False
    dialog.reject()


def test_shortcut_interaction_marks_learning_complete(tmp_path: Path) -> None:
    _application()
    called: list[bool] = []
    dialog = OnboardingDialog(
        FakeManager(),
        Settings(data_dir=tmp_path),
        question_callback=lambda: called.append(True),
    )
    dialog._set_page(4)
    dialog.onboarding_shortcut.setKeySequence(QKeySequence("Alt+F8"))
    assert dialog.state.question_shortcut == "Alt+F8"
    dialog._shortcut.activated.emit()
    assert called == [True]
    assert dialog.state.shortcuts_completed is True
    assert dialog.next_button.isEnabled()
    assert "快捷键测试完成" in dialog.shortcut_status.text()
    dialog._set_page(3)
    dialog.capsule_preview_question.click()
    assert "提问入口已触发" in dialog.capsule_preview_status.text()
    dialog.reject()


def test_shortcut_page_can_focus_main_question_input(tmp_path: Path) -> None:
    _application()
    window = MainWindow(Settings(data_dir=tmp_path))
    dialog = OnboardingDialog(
        window.manager,
        window.settings,
        parent=window,
        question_callback=window._focus_question,
    )
    window.show()
    dialog.show()
    dialog._set_page(4)
    dialog._shortcut_triggered()
    QApplication.processEvents()
    assert window.question.hasFocus()
    dialog.reject()
    window.close()


def test_whisper_benchmark_success_and_failure_keep_status_local(tmp_path: Path) -> None:
    _application()
    dialog = OnboardingDialog(FakeManager(), Settings(data_dir=tmp_path))
    dialog._set_page(2)
    dialog._onboarding_task = "whisper"
    dialog._whisper_benchmark_succeeded(SimpleNamespace(elapsed_seconds=1.5, realtime_factor=0.8))
    assert dialog.state.whisper_completed is True
    assert "样本测试完成" in dialog.whisper_status.text()
    provider_status = dialog.provider_status.text()
    dialog._onboarding_task = "whisper"
    dialog._task_failed("whisper", "模型文件不可用")
    assert dialog.provider_status.text() == provider_status
    assert "样本测试失败" in dialog.whisper_status.text()
    dialog.reject()


def test_existing_whisper_profile_change_is_saved_on_next(tmp_path: Path) -> None:
    _application()
    manager = FakeManager(
        whisper_settings=replace(
            PROFILE_PRESETS[WhisperProfile.ACCURATE].settings,
            first_run_completed=True,
        )
    )
    dialog = OnboardingDialog(manager, Settings(data_dir=tmp_path))
    dialog._set_page(2)
    dialog.onboarding_whisper_profile.setCurrentIndex(
        dialog.onboarding_whisper_profile.findData(WhisperProfile.BALANCED.value)
    )
    dialog._next_page()
    assert manager.whisper_settings.profile == WhisperProfile.BALANCED
    assert manager.saved_whisper is True
    dialog.reject()


def test_failed_provider_test_restores_unpersisted_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    _application()

    class FailingManager(FakeManager):
        def test_provider(self) -> str:
            raise RuntimeError("连接不可用")

    class ImmediateThread:
        def __init__(self, target, **_kwargs) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    manager = FailingManager()
    original = manager.provider_settings
    monkeypatch.setattr("jingzhi.ui.threading.Thread", ImmediateThread)
    dialog = OnboardingDialog(manager, Settings(data_dir=tmp_path))
    dialog._test_provider(create_new=True)
    assert manager.provider_settings == original
    assert dialog._onboarding_task is None
    assert dialog.state.provider_completed is False
    assert "连接不可用" in dialog.provider_status.text()
    dialog.reject()


def test_failed_provider_operation_keeps_step_retryable(tmp_path: Path) -> None:
    _application()
    dialog = OnboardingDialog(FakeManager(), Settings(data_dir=tmp_path))
    dialog._task_failed("provider", "网络不可用")
    assert dialog.pages.currentIndex() == 0
    dialog._set_page(1)
    assert dialog.provider_test_button.isEnabled()
    assert dialog.provider_create_button.isEnabled()
    assert dialog.state.provider_completed is False
    assert "网络不可用" in dialog.provider_status.text()
    dialog.reject()


def test_recording_capsule_start_button_calls_start_without_boolean_selection(
    tmp_path: Path,
) -> None:
    _application()
    window = MainWindow(Settings(data_dir=tmp_path))
    selections: list[object] = []
    window._start = lambda selection=None: selections.append(selection) or False
    window.start_button.click()
    assert selections == [None]
    window.close()


def test_main_window_keeps_onboarding_open_when_start_fails(tmp_path: Path) -> None:
    _application()
    window = MainWindow(Settings(data_dir=tmp_path))
    dialog = OnboardingDialog(
        window.manager,
        window.settings,
        parent=window,
        state_store=window._onboarding_store,
    )
    window._start = lambda selection=None: False
    window._onboarding_dialog = dialog
    dialog.finished.connect(window._onboarding_finished)
    dialog.accept()
    assert window._onboarding_dialog is dialog
    assert dialog.isVisible()
    assert "启动失败" in dialog.shortcut_status.text()
    assert window._onboarding_store.load().completed is False
    dialog.reject()
    window._onboarding_dialog = None
    window.close()


def test_main_window_marks_onboarding_complete_only_after_start_succeeds(tmp_path: Path) -> None:
    _application()
    window = MainWindow(Settings(data_dir=tmp_path))
    dialog = OnboardingDialog(
        window.manager,
        window.settings,
        parent=window,
        state_store=window._onboarding_store,
    )
    selection = RecordingSelection(("display:1",), None, None, 30)
    dialog.state = replace(
        OnboardingState(),
        step="shortcuts",
        privacy_acknowledged=True,
        provider_skipped=True,
        whisper_skipped=True,
        recording_confirmed=True,
        shortcuts_completed=True,
    )
    dialog._recording_selection = selection
    dialog.state_store.save(dialog.state)
    provider = window.manager.provider_settings
    created_connection = replace(
        provider.connections[0],
        id="created",
        name="引导连接",
        base_url="https://example.test",
    )
    roles = tuple(
        replace(role, connection_id="created") if role.name == RoleName.UTILITY else role
        for role in provider.roles
    )
    window.manager.configure_provider(
        replace(
            provider,
            connections=(created_connection, *provider.connections),
            roles=roles,
        )
    )
    window._reload_provider_settings_from_manager()
    assert window._provider_connections[0].id == "created"
    assert window.connection_selector.count() == len(window._provider_connections)
    assert window.role_connection_inputs[RoleName.UTILITY].currentData() == "created"
    started: list[RecordingSelection | None] = []
    window._start = lambda selection=None: started.append(selection) or True
    window._onboarding_dialog = dialog
    dialog.finished.connect(window._onboarding_finished)
    dialog.accept()
    assert started == [selection]
    assert window._onboarding_store.load().completed is True
    window.close()


def test_provider_test_success_marks_step_and_updates_roles(tmp_path: Path) -> None:
    _application()
    manager = FakeManager()
    dialog = OnboardingDialog(manager, Settings(data_dir=tmp_path))
    dialog.onboarding_provider_model.setText("model-under-test")
    manager.configure_provider(dialog._provider_settings_from_form())
    dialog._onboarding_task = "provider"
    dialog._provider_test_succeeded("ok")
    utility = next(
        role for role in manager.provider_settings.roles if role.name == RoleName.UTILITY
    )
    assert utility.model == "model-under-test"
    created = dialog._provider_settings_from_form(create_new=True)
    assert len(created.connections) == 2
    assert created.connections[0].id != created.connections[1].id
    assert (
        next(role for role in created.roles if role.name == RoleName.UTILITY).connection_id
        == created.connections[0].id
    )
    assert manager.saved_provider is True
    assert dialog.state.provider_completed is True
    dialog.reject()
