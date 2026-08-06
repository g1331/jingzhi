import os
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import keyring
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QWidget,
)

from jingzhi.application import JingzhiApplicationService
from jingzhi.config import Settings
from jingzhi.cross_session import CrossSessionSynthesisError, CrossSessionSynthesisPreview
from jingzhi.database import (
    CrossSessionEvidenceRecord,
    CrossSessionSearchResult,
    Database,
)
from jingzhi.model_roles import RoleName
from jingzhi.provider_settings import ProviderSettingsStore, default_saved_settings
from jingzhi.ui import CrossSessionSynthesisDialog, EvidenceButton, MainWindow


class NoHardwareRecorder:
    is_recording = False

    def start(self, title, **_kwargs):
        raise AssertionError(f"Unexpected hardware start for {title}")

    def stop(self):
        return None


class CrossSessionDialogService:
    def __init__(self) -> None:
        self.preview_calls: list[tuple[str, tuple[str, ...]]] = []
        self.failed_items: tuple[object, ...] = ()
        self.saved_evidence: dict[int, tuple[object, ...]] = {}
        self.results = [
            CrossSessionSearchResult(
                "answer-version:1",
                "session-1",
                "会话一",
                "answer",
                1,
                "问答",
                1_000,
                1_000,
                None,
                "问题\n答案关键词",
                "答案关键词",
            ),
            CrossSessionSearchResult(
                "answer-version:4",
                "session-2",
                "会话二",
                "answer",
                4,
                "问答",
                2_000,
                2_000,
                None,
                "另一个答案关键词",
                "另一个答案关键词",
            ),
        ]
        self.candidates = [
            CrossSessionEvidenceRecord(
                "answer-version:1",
                "session-1",
                "会话一",
                "answer",
                "问答",
                1_000,
                1_000,
                "答案关键词",
                None,
                answer_version_id=1,
            ),
            CrossSessionEvidenceRecord(
                "transcript-version:2",
                "session-1",
                "会话一",
                "transcript",
                "麦克风",
                500,
                1_500,
                "字幕关键词",
                None,
                transcript_version_id=2,
            ),
            CrossSessionEvidenceRecord(
                "frame:3",
                "session-1",
                "会话一",
                "frame",
                "display:1",
                1_000,
                1_000,
                None,
                Path("frame.webp"),
                frame_id=3,
            ),
            CrossSessionEvidenceRecord(
                "answer-version:4",
                "session-2",
                "会话二",
                "answer",
                "问答",
                2_000,
                2_000,
                "另一个答案关键词",
                None,
                answer_version_id=4,
            ),
        ]

    def failed_cross_session_syntheses(self, *, limit: int = 10):
        del limit
        return self.failed_items

    def cross_session_synthesis_evidence(self, synthesis_id: int):
        return self.saved_evidence.get(synthesis_id, ())

    def cross_session_search(self, query: str, *, limit: int = 50):
        del limit
        return [] if query == "无结果" else self.results

    def cross_session_evidence_candidates(self, stable_ids: tuple[str, ...]):
        requested = set(stable_ids)
        if "answer-version:1" in requested:
            return self.candidates[:3]
        if "answer-version:4" in requested:
            return [self.candidates[3]]
        return []

    def cross_session_synthesis_preview(self, question: str, stable_ids: tuple[str, ...]):
        self.preview_calls.append((question, stable_ids))
        return CrossSessionSynthesisPreview(
            question=question,
            stable_ids=stable_ids,
            evidence_count=len(stable_ids),
            character_count=10,
            frame_count=0,
            connection_name="连接",
            model="模型",
            reasoning_level="deep",
            can_synthesize=bool(question and stable_ids),
        )


class VisualStateService(JingzhiApplicationService):
    def __init__(self, *args, cited_frame_id: int, cited_transcript_id: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cited_frame_id = cited_frame_id
        self.cited_transcript_id = cited_transcript_id

    def open_session(self, *args, **kwargs):
        timeline = super().open_session(*args, **kwargs)
        transcripts = tuple(
            replace(item, correction_state="corrected")
            if item.id == self.cited_transcript_id
            else item
            for item in timeline.transcripts
        )
        return replace(
            timeline,
            transcripts=transcripts,
            answer_frame_ids=frozenset({self.cited_frame_id}),
            answer_transcript_ids=frozenset({self.cited_transcript_id}),
        )


def test_cross_session_dialog_requires_explicit_evidence_selection_and_exposes_navigation() -> None:
    application = QApplication.instance() or QApplication([])
    navigated: list[CrossSessionEvidenceRecord] = []
    dialog = CrossSessionSynthesisDialog(
        CrossSessionDialogService(), navigate_callback=navigated.append
    )

    dialog.query_input.setText("关键词")
    dialog._search()
    application.processEvents()

    assert dialog.result_list.count() == 2
    assert dialog.evidence_list.count() == 3
    assert dialog.synthesize_button.isEnabled() is False

    first = dialog.evidence_list.item(0)
    first.setCheckState(Qt.CheckState.Checked)
    dialog.question_input.setText("比较授权证据")
    application.processEvents()

    assert dialog.synthesize_button.isEnabled() is True
    assert dialog.service.preview_calls[-1] == ("比较授权证据", ("answer-version:1",))
    dialog.result_list.setCurrentRow(1)
    second = dialog.evidence_list.item(0)
    second.setCheckState(Qt.CheckState.Checked)
    application.processEvents()
    assert dialog.service.preview_calls[-1] == (
        "比较授权证据",
        ("answer-version:1", "answer-version:4"),
    )
    dialog.result_list.setCurrentRow(0)
    dialog.output.show()
    dialog.navigate_button.setEnabled(True)
    dialog.query_input.setText("无结果")
    dialog._search()
    assert dialog._selected_ids == set()
    assert dialog.synthesize_button.isEnabled() is False
    assert dialog.output.isHidden() is True
    assert dialog.navigate_button.isEnabled() is False
    dialog.query_input.setText("关键词")
    dialog._search()
    dialog.evidence_list.setCurrentRow(1)
    dialog._navigate_current()
    assert navigated[-1].stable_id == "transcript-version:2"
    dialog.close()


def test_main_window_navigates_cross_session_frame_to_timeline(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "cross-navigation.sqlite3")
    session_id = database.create_session("可跳转会话", "2026-08-01T00:00:00+00:00")
    frame_path = tmp_path / "navigation.webp"
    Image.new("RGB", (320, 180), "navy").save(frame_path)
    frame_id = database.add_frame(
        session_id,
        45_000,
        frame_path,
        "navigation-hash",
        (320, 180),
        source_id="display:2",
    )
    database.finish_session(session_id, "2026-08-01T00:01:00+00:00", "complete")
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    application.processEvents()

    window._navigate_cross_session_evidence(
        CrossSessionEvidenceRecord(
            f"frame:{frame_id}",
            session_id,
            "可跳转会话",
            "frame",
            "display:2",
            45_000,
            45_000,
            None,
            frame_path,
            frame_id=frame_id,
        )
    )
    application.processEvents()

    assert window._selected_session_id == session_id
    assert window._selected_frame is not None
    assert window._selected_frame.id == frame_id
    assert window._timeline is not None
    assert window._timeline.window_start_ms <= 45_000 <= window._timeline.window_end_ms
    window.close()


def test_main_window_navigates_historical_cross_session_transcript_version(
    tmp_path: Path,
) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "historical-navigation.sqlite3")
    session_id = database.create_session("历史字幕会话", "2026-08-01T00:00:00+00:00")
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 5_000, tmp_path / "audio.wav")
    segment_id = database.add_transcript(
        session_id, chunk_id, "system", 1_000, 2_000, "原始版本", "zh", 0.9
    )
    original_version_id = database.transcript_versions(segment_id)[0].id
    edited_version_id = database.add_transcript_version(segment_id, "user_edit", "当前编辑版本")
    assert edited_version_id is not None
    database.finish_session(session_id, "2026-08-01T00:01:00+00:00", "complete")
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    application.processEvents()

    window._navigate_cross_session_evidence(
        CrossSessionEvidenceRecord(
            f"transcript-version:{original_version_id}",
            session_id,
            "历史字幕会话",
            "transcript",
            "system",
            1_000,
            2_000,
            "原始版本",
            None,
            transcript_version_id=original_version_id,
        )
    )
    application.processEvents()

    assert window._selected_transcript is not None
    assert window._selected_transcript.version_id == original_version_id
    assert window._selected_transcript.text == "原始版本"
    assert edited_version_id != original_version_id
    window.close()


def test_cross_session_dialog_exposes_retry_for_persisted_failure() -> None:
    application = QApplication.instance() or QApplication([])
    service = CrossSessionDialogService()
    service.failed_items = (SimpleNamespace(id=42),)
    service.saved_evidence[42] = (SimpleNamespace(stable_id="answer-version:1"),)
    dialog = CrossSessionSynthesisDialog(service)

    assert dialog._failed_synthesis_id == 42
    assert dialog._selected_stable_ids() == ("answer-version:1",)
    assert dialog.retry_button.isHidden() is False
    dialog.query_input.setText("关键词")
    dialog._search()
    assert dialog._failed_synthesis_id is None
    assert dialog.retry_button.isHidden() is True
    dialog._show_synthesis_error(CrossSessionSynthesisError("模型失败", 43))
    application.processEvents()

    assert dialog.retry_button.isHidden() is False
    assert dialog.retry_button.isEnabled() is True
    dialog.close()


def _frame_buttons(window: MainWindow) -> list[QPushButton]:
    return [
        button
        for button in window.findChildren(QPushButton)
        if button.objectName().startswith("keyframe-")
    ]


def _contains_rgb(image, target: tuple[int, int, int], tolerance: int = 4) -> bool:
    target_red, target_green, target_blue = target
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if (
                abs(color.red() - target_red) <= tolerance
                and abs(color.green() - target_green) <= tolerance
                and abs(color.blue() - target_blue) <= tolerance
            ):
                return True
    return False


def test_clearing_timeline_layout_hides_widgets_before_detaching_them() -> None:
    class DetachTrackingButton(QPushButton):
        visible_when_detached: bool | None = None

        def setParent(self, parent) -> None:  # type: ignore[no-untyped-def]
            if parent is None:
                self.visible_when_detached = self.isVisible()
            super().setParent(parent)

    application = QApplication.instance() or QApplication([])
    parent = QWidget()
    layout = QHBoxLayout(parent)
    button = DetachTrackingButton("旧关键帧")
    layout.addWidget(button)
    parent.show()
    application.processEvents()

    MainWindow._clear_layout(layout)

    assert button.visible_when_detached is False
    assert button.isVisible() is False
    parent.close()


def test_html_warning_is_compacted_and_does_not_change_window_width(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(Settings(data_dir=tmp_path))
    initial_width = window.width()

    window._show_worker_warning("<!DOCTYPE html><html>" + "unbroken" * 2_000 + "</html>")
    application.processEvents()

    assert "网页源码" in window.notice_text.text()
    assert len(window.notice_text.text()) < 120
    assert window.width() == initial_width
    window.manager.save_provider = lambda: None
    window.close()


def test_pause_control_starts_disabled_when_idle(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(Settings(data_dir=tmp_path))
    application.processEvents()

    assert window.pause_button.isEnabled() is False
    window.close()


def test_archive_controls_require_a_session_for_export_and_expose_backup_actions(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(Settings(data_dir=tmp_path))
    application.processEvents()

    assert window.session_export_button.isEnabled() is False
    assert window.backup_button.isEnabled() is True
    assert window.restore_backup_button.isEnabled() is True

    session_id = window.service.database.create_session("可导出会话", "2026-08-04T10:00:00+00:00")
    window.service.database.finish_session(session_id, "2026-08-04T10:01:00+00:00", "complete")
    window._refresh_sessions(session_id)
    application.processEvents()

    assert window.session_export_button.isEnabled() is True
    window.close()


def test_start_waits_for_interrupted_workers_to_exit(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(Settings(data_dir=tmp_path))
    window.manager.session_id = "interrupted-session"
    application.processEvents()
    window.service.session_storage_busy_reason = lambda _session_id: "采集线程仍在写入"

    window._refresh_recording_status()
    assert window.start_button.isEnabled() is False

    window.service.session_storage_busy_reason = lambda _session_id: None
    window._stop_in_flight = True
    window._refresh_recording_status()
    assert window.start_button.isEnabled() is False

    window._stop_in_flight = False
    window._refresh_recording_status()
    assert window.start_button.isEnabled() is True
    window.close()


def test_unconfirmed_source_event_is_replayed_on_session_open(tmp_path, monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "source-event-ui.sqlite3")
    session_id = database.create_session("待确认缺口", "2026-08-04T00:00:00+00:00")
    event_id = database.record_source_event(
        session_id,
        "microphone",
        "device_unavailable",
        1_000,
        3_000,
        "设备已拔出",
    )
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())
    monkeypatch.setattr(
        "jingzhi.ui.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    application.processEvents()

    events = database.timeline_events(session_id, 0, 10_000)
    assert [(event.kind, event.source_event_id) for event in events] == [("data_gap", event_id)]
    window.close()


def test_saving_provider_keeps_role_models(tmp_path, monkeypatch) -> None:
    stored_secret: dict[str, str] = {}
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda _service, _username, password: stored_secret.update(value=password),
    )
    monkeypatch.setattr(
        keyring,
        "get_password",
        lambda _service, _username: stored_secret.get("value"),
    )
    monkeypatch.setattr(keyring, "delete_password", lambda _service, _username: None)
    application = QApplication.instance() or QApplication([])
    window = MainWindow(
        Settings(data_dir=tmp_path, provider_settings=default_saved_settings("main-model"))
    )
    window.role_model_inputs[RoleName.INSTANT_ANSWER].setText("main-model")
    window.role_model_inputs[RoleName.TRANSCRIPT_CORRECTION].setText("luna")

    window._save_provider()
    application.processEvents()

    saved = ProviderSettingsStore(tmp_path).load()
    roles = {role.name: role for role in saved.roles}
    assert roles[RoleName.INSTANT_ANSWER].model == "main-model"
    assert roles[RoleName.TRANSCRIPT_CORRECTION].model == "luna"
    window.manager.save_provider = lambda: None
    window.close()


def test_cross_connection_fallback_requires_explicit_authorization(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(Settings(data_dir=tmp_path, provider_settings=default_saved_settings()))
    window._add_provider_connection()
    primary_id = window._provider_connections[0].id
    backup_id = window._provider_connections[-1].id
    role = RoleName.INSTANT_ANSWER
    first_fallback = window.role_fallback_connection_inputs[role]
    first_fallback.setCurrentIndex(first_fallback.findData(primary_id))
    window.role_fallback_model_inputs[role].setText("answer-small")
    second_fallback = window.role_second_fallback_connection_inputs[role]
    second_fallback.setCurrentIndex(second_fallback.findData(backup_id))
    window.role_second_fallback_model_inputs[role].setText("answer-backup")
    window.role_second_cross_auth_checks[role].setChecked(False)
    assert window._configure_provider_from_form() is True
    configured = window.manager.provider_settings
    instant = next(item for item in configured.roles if item.name == role)
    assert [item.model for item in instant.fallbacks] == ["answer-small", "answer-backup"]
    assert instant.fallbacks[1].cross_connection_authorized is False
    window.role_second_cross_auth_checks[role].setChecked(True)
    assert window._configure_provider_from_form() is True
    instant = next(item for item in window.manager.provider_settings.roles if item.name == role)
    assert instant.fallbacks[1].cross_connection_authorized is True
    window.manager.save_provider = lambda: None
    window.close()
    application.processEvents()


def test_session_selection_thumbnail_zoom_and_detail_switching(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "test.sqlite3")
    session_id = database.create_session("界面验收会话", "2026-08-02T09:00:00+00:00")
    other_id = database.create_session("较早会话", "2026-08-01T09:00:00+00:00")
    database.finish_session(other_id, "2026-08-01T09:01:00+00:00", "complete")
    frame_ids = []
    for index, (ts_ms, source, color) in enumerate(
        [(15_000, "display:1", "white"), (320_000, "display:2", "navy")], start=1
    ):
        image_path = tmp_path / f"frame-{index}.webp"
        Image.new("RGB", (640, 360), color).save(image_path)
        frame_ids.append(
            database.add_frame(
                session_id,
                ts_ms,
                image_path,
                f"hash-{index}",
                (640, 360),
                source_id=source,
            )
        )
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 60_000, tmp_path / "audio.wav")
    transcript_id = database.add_transcript(
        session_id,
        chunk_id,
        "system",
        20_000,
        28_000,
        "左右极限不同，因此这里不连续。",
        "zh",
        0.9,
    )
    database.add_question(session_id, 25_000, "为什么不连续？", None, 0, 25_000)
    database.finish_session(session_id, "2026-08-02T09:06:00+00:00", "complete")
    service = VisualStateService(
        database,
        recorder=NoHardwareRecorder(),
        now=lambda: datetime(2026, 8, 2, 9, 6, tzinfo=UTC),
        cited_frame_id=frame_ids[0],
        cited_transcript_id=transcript_id,
    )
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.resize(1280, 720)
    window.show()
    application.processEvents()

    library = window.findChild(QListWidget, "sessionLibrary")
    assert library is not None
    assert library.count() == 2
    matching_items = [
        library.item(index)
        for index in range(library.count())
        if library.item(index).data(Qt.ItemDataRole.UserRole) == session_id
    ]
    library.setCurrentItem(matching_items[0])
    application.processEvents()
    assert len(_frame_buttons(window)) == 2

    zoom = window.findChild(QPushButton, "zoom-1-minute")
    assert zoom is not None
    zoom.click()
    application.processEvents()
    assert zoom.isChecked()
    assert len(_frame_buttons(window)) == 1

    first_frame_button = _frame_buttons(window)[0]
    assert first_frame_button.property("cited") is True
    first_frame_button.click()
    application.processEvents()
    assert first_frame_button.property("selected") is True
    detail_image = window.findChild(QLabel, "evidenceImage")
    detail_metadata = window.findChild(QLabel, "evidenceMetadata")
    assert detail_image is not None and detail_image.pixmap() is not None
    assert detail_metadata is not None
    assert "display:1" in detail_metadata.text()
    assert "00:15" in detail_metadata.text()

    transcript_button = window.findChild(QPushButton, f"transcript-{transcript_id}")
    assert transcript_button is not None
    assert transcript_button.property("cited") is True
    assert "已校订" in transcript_button.text()
    transcript_button.click()
    application.processEvents()
    assert "system" in detail_metadata.text()
    version_detail = window.evidence_version.text()
    assert "已校订" in version_detail
    assert "Whisper 原文" in version_detail
    assert "Q 00:25" in window.event_text.text()
    assert "2026-08-02T09:00" not in window.event_text.text()

    navigator = window.findChild(QSlider, "timelineNavigator")
    assert navigator is not None and navigator.isEnabled()
    navigator.setValue(300)
    application.processEvents()
    assert [button.text() for button in _frame_buttons(window)] == ["05:20\ndisplay:2"]

    shorter_item = next(
        library.item(index)
        for index in range(library.count())
        if library.item(index).data(Qt.ItemDataRole.UserRole) == other_id
    )
    library.setCurrentItem(shorter_item)
    application.processEvents()
    assert navigator.value() == 0
    assert window._timeline is not None and window._timeline.window_start_ms == 0

    window.capsule_ask_button.click()
    application.processEvents()
    assert window.question.hasFocus()
    assert window.pause_button.isVisible()

    library_panel = window.findChild(QWidget, "libraryPanel")
    detail_panel = window.findChild(QWidget, "detailPanel")
    keyframe_track = window.findChild(QWidget, "keyframeTrack")
    transcript_track = window.findChild(QWidget, "transcriptTrack")
    event_track = window.findChild(QWidget, "eventTrack")
    assert library_panel is not None and 190 <= library_panel.width() <= 240
    assert detail_panel is not None and 250 <= detail_panel.width() <= 310
    assert all(
        widget is not None and widget.isVisible()
        for widget in (keyframe_track, transcript_track, event_track)
    )

    window.close()


def test_answer_selection_updates_exact_evidence_across_zoom_and_unavailable_history(
    tmp_path: Path,
) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "answer-selection.sqlite3")
    session_id = database.create_session("问答证据", "2026-08-03T09:00:00+00:00")

    frame_ids: list[int] = []
    transcript_ids: list[int] = []
    transcript_version_ids: list[int] = []
    for index, ts_ms in enumerate((15_000, 320_000), start=1):
        image_path = tmp_path / f"answer-frame-{index}.webp"
        Image.new("RGB", (320, 180), "white" if index == 1 else "navy").save(image_path)
        frame_ids.append(
            database.add_frame(
                session_id,
                ts_ms,
                image_path,
                f"answer-hash-{index}",
                (320, 180),
                source_id=f"display:{index}",
            )
        )
        chunk_id = database.add_audio_chunk(
            session_id,
            "system",
            ts_ms,
            ts_ms + 8_000,
            tmp_path / f"answer-audio-{index}.flac",
        )
        transcript_id = database.add_transcript(
            session_id,
            chunk_id,
            "system",
            ts_ms + 1_000,
            ts_ms + 6_000,
            f"回答 {index} 使用的原始字幕",
            "zh",
            0.9,
        )
        transcript_ids.append(transcript_id)
        transcript_version_ids.append(database.transcript_versions(transcript_id)[0].id)

    question_ids: list[int] = []

    answer_ids: list[int] = []
    for index in range(2):
        question_id = database.create_question(
            session_id,
            25_000 + index * 310_000,
            f"第 {index + 1} 个问题",
            0,
            25_000 + index * 310_000,
        )
        question_ids.append(question_id)

        answer = database.record_answer_version(
            question_id,
            model="answer-model",
            connection_json=None,
            request_status="succeeded",
            request_id=None,
            answer=f"第 {index + 1} 个回答",
            error=None,
            evidence_state="exact",
            evidence=[
                {
                    "stable_id": f"transcript-version:{transcript_version_ids[index]}",
                    "kind": "transcript",
                    "source": "system",
                    "start_ms": 16_000 + index * 305_000,
                    "end_ms": 21_000 + index * 305_000,
                    "transcript_version_id": transcript_version_ids[index],
                    "content_text": f"回答 {index + 1} 使用的原始字幕",
                },
                {
                    "stable_id": f"frame:{frame_ids[index]}",
                    "kind": "frame",
                    "source": f"display:{index + 1}",
                    "start_ms": 15_000 + index * 305_000,
                    "end_ms": 15_000 + index * 305_000,
                    "frame_id": frame_ids[index],
                    "resource_path": str(tmp_path / f"answer-frame-{index + 1}.webp"),
                },
            ],
        )
        answer_ids.append(answer.id)

    legacy_question_id = database.create_question(session_id, 350_000, "历史问题", 340_000, 350_000)
    unavailable = database.record_answer_version(
        legacy_question_id,
        model=None,
        connection_json=None,
        request_status="succeeded",
        request_id=None,
        answer="历史回答",
        error=None,
        evidence_state="unavailable",
        evidence=[],
    )
    database.add_transcript_version(
        transcript_ids[0], "user_edit", "后来编辑但不应替代旧回答证据的字幕"
    )
    database.finish_session(session_id, "2026-08-03T09:06:00+00:00", "complete")

    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.show()
    application.processEvents()

    selector = window.findChild(QComboBox, "answerSelector")
    status = window.findChild(QLabel, "answerEvidenceStatus")
    assert selector is not None and status is not None
    assert selector.count() == 3

    selector.setCurrentIndex(selector.findData(answer_ids[0]))
    application.processEvents()
    first_frame = window.findChild(QPushButton, f"keyframe-{frame_ids[0]}")
    second_frame = window.findChild(QPushButton, f"keyframe-{frame_ids[1]}")
    first_transcript = window.findChild(QPushButton, f"transcript-{transcript_ids[0]}")
    second_transcript = window.findChild(QPushButton, f"transcript-{transcript_ids[1]}")
    assert first_frame.property("cited") is True
    assert first_transcript.property("cited") is True
    assert second_frame.property("cited") is False
    assert second_transcript.property("cited") is False
    assert "回答 1 使用的原始字幕" in first_transcript.text()
    assert status.property("state") == "exact"
    assert status.text() == "会话证据 · 1 张关键帧 · 1 条字幕 · 00:15–00:21"
    assert status.toolTip() == (
        f"稳定证据标识：\ntranscript-version:{transcript_version_ids[0]}\nframe:{frame_ids[0]}"
    )
    assert "## 无法确认" in window.output.raw.toPlainText()
    assert "模型回答未标明依据边界" in window.output.raw.toPlainText()

    selector.setCurrentIndex(selector.findData(answer_ids[1]))
    application.processEvents()
    first_frame = window.findChild(QPushButton, f"keyframe-{frame_ids[0]}")
    second_frame = window.findChild(QPushButton, f"keyframe-{frame_ids[1]}")
    assert first_frame.property("cited") is False
    assert second_frame.property("cited") is True
    first_transcript = window.findChild(QPushButton, f"transcript-{transcript_ids[0]}")
    second_transcript = window.findChild(QPushButton, f"transcript-{transcript_ids[1]}")
    assert first_transcript.property("cited") is False
    assert second_transcript.property("cited") is True

    window.findChild(QPushButton, "zoom-1-minute").click()
    application.processEvents()
    assert window.findChild(QPushButton, f"keyframe-{frame_ids[0]}").property("cited") is False
    assert (
        window.findChild(QPushButton, f"transcript-{transcript_ids[0]}").property("cited") is False
    )

    navigator = window.findChild(QSlider, "timelineNavigator")
    navigator.setValue(300)
    application.processEvents()
    assert window.findChild(QPushButton, f"keyframe-{frame_ids[1]}").property("cited") is True
    second_transcript = window.findChild(QPushButton, f"transcript-{transcript_ids[1]}")
    assert second_transcript.property("cited") is True
    assert "回答 2 使用的原始字幕" in second_transcript.text()

    selector = window.findChild(QComboBox, "answerSelector")
    selector.setCurrentIndex(selector.findData(unavailable.id))
    application.processEvents()
    assert all(button.property("cited") is False for button in _frame_buttons(window))
    visible_transcripts = [
        button
        for button in window.findChildren(QPushButton)
        if button.objectName().startswith("transcript-")
    ]
    assert all(button.property("cited") is False for button in visible_transcripts)

    assert status.property("state") == "unavailable"
    assert "不可恢复" in status.text()

    new_answer = database.record_answer_version(
        question_ids[0],
        model="answer-model",
        connection_json=None,
        request_status="succeeded",
        request_id=None,
        answer="刚完成的回答",
        error=None,
        evidence_state="exact",
        evidence=[
            {
                "stable_id": f"frame:{frame_ids[1]}",
                "kind": "frame",
                "source": "display:2",
                "start_ms": 320_000,
                "end_ms": 320_000,
                "frame_id": frame_ids[1],
                "resource_path": str(tmp_path / "answer-frame-2.webp"),
            }
        ],
    )
    window._show_answer(question_ids[0], "刚完成的回答")

    application.processEvents()
    assert selector.count() == 4
    assert selector.currentData() == new_answer.id
    assert window.findChild(QPushButton, f"keyframe-{frame_ids[1]}").property("cited") is True
    window.close()


def test_answer_evidence_entries_navigate_to_exact_frame_and_transcript(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "navigation.sqlite3")
    session_id = database.create_session("证据定位", "2026-08-04T09:00:00+00:00")
    frame_dir = tmp_path / "sessions" / session_id / "frames"
    frame_dir.mkdir(parents=True)
    frame_path = frame_dir / "frame.webp"
    Image.new("RGB", (640, 360), "navy").save(frame_path)
    frame_id = database.add_frame(
        session_id,
        320_000,
        frame_path,
        "navigation-frame",
        (640, 360),
        source_id="display:2",
    )
    chunk_id = database.add_audio_chunk(
        session_id,
        "system",
        321_000,
        329_000,
        tmp_path / "sessions" / session_id / "audio" / "chunk.flac",
    )
    segment_id = database.add_transcript(
        session_id,
        chunk_id,
        "system",
        322_000,
        327_000,
        "换入便量",
        "zh",
        -0.2,
    )
    corrected_version_id = database.add_transcript_version(
        segment_id, "correction", "换入变量", model="correction-small"
    )
    question_id = database.create_question(session_id, 330_000, "这里说了什么？", 300_000, 330_000)
    answer = database.record_answer_version(
        question_id,
        model="answer-model",
        connection_json=None,
        request_status="succeeded",
        request_id=None,
        answer="回答引用了两项证据。",
        error=None,
        evidence_state="exact",
        evidence=[
            {
                "stable_id": f"frame:{frame_id}",
                "kind": "frame",
                "source": "display:2",
                "start_ms": 320_000,
                "end_ms": 320_000,
                "frame_id": frame_id,
                "resource_path": str(frame_path),
            },
            {
                "stable_id": f"transcript-version:{corrected_version_id}",
                "kind": "transcript",
                "source": "system",
                "start_ms": 322_000,
                "end_ms": 327_000,
                "transcript_version_id": corrected_version_id,
                "content_text": "换入变量",
            },
        ],
    )
    database.finish_session(session_id, "2026-08-04T09:06:00+00:00", "complete")

    window = MainWindow(
        Settings(data_dir=tmp_path),
        service=JingzhiApplicationService(database, recorder=NoHardwareRecorder()),
    )
    window.show()
    application.processEvents()

    frame_entry = window.findChild(QPushButton, "answer-evidence-0")
    transcript_entry = window.findChild(QPushButton, "answer-evidence-1")
    assert frame_entry is not None and transcript_entry is not None
    assert frame_entry.property("stableId") == f"frame:{frame_id}"
    assert transcript_entry.property("stableId") == f"transcript-version:{corrected_version_id}"

    frame_entry.click()
    application.processEvents()
    assert window._zoom_key == "1-minute"
    assert window._timeline is not None
    assert window._timeline.window_start_ms <= 320_000 <= window._timeline.window_end_ms
    assert window.timeline_navigator.value() == window._timeline.window_start_ms // 1000
    selected_frame = window.findChild(QPushButton, f"keyframe-{frame_id}")
    assert selected_frame is not None and selected_frame.property("selected") is True
    assert window.evidence_image.pixmap() is not None
    assert "display:2" in window.evidence_metadata.text()

    transcript_entry = window.findChild(QPushButton, "answer-evidence-1")
    assert transcript_entry is not None
    transcript_entry.click()
    application.processEvents()
    selected_transcript = window.findChild(QPushButton, f"transcript-{segment_id}")
    assert selected_transcript is not None and selected_transcript.property("selected") is True
    assert "原文：换入便量" in window.evidence_version.text()
    assert "校订文：换入变量" in window.evidence_version.text()
    assert window.transcript_diff_button.isVisible()
    window.transcript_diff_button.click()
    assert "[-便-]{+变+}" in window.evidence_version.text()
    assert window.answer_selector.currentData() == answer.id
    window.close()


def test_answer_evidence_navigation_degrades_for_missing_and_unsafe_targets(
    tmp_path: Path,
) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "unsafe-navigation.sqlite3")
    session_id = database.create_session("安全定位", "2026-08-04T09:00:00+00:00")
    other_session_id = database.create_session("其他会话", "2026-08-04T08:00:00+00:00")
    missing_path = tmp_path / "sessions" / session_id / "frames" / "missing.webp"
    missing_frame_id = database.add_frame(
        session_id,
        120_000,
        missing_path,
        "missing-frame",
        (640, 360),
        source_id="display:1",
    )
    other_path = tmp_path / "sessions" / other_session_id / "frames" / "other.webp"
    other_path.parent.mkdir(parents=True)
    Image.new("RGB", (320, 180), "white").save(other_path)
    other_frame_id = database.add_frame(
        other_session_id,
        240_000,
        other_path,
        "other-frame",
        (320, 180),
        source_id="display:9",
    )
    outside_path = tmp_path / "outside.webp"
    Image.new("RGB", (320, 180), "red").save(outside_path)
    outside_frame_id = database.add_frame(
        session_id,
        260_000,
        outside_path,
        "outside-frame",
        (320, 180),
        source_id="display:1",
    )
    question_id = database.create_question(session_id, 300_000, "证据安全吗？", 0, 300_000)
    database.record_answer_version(
        question_id,
        model="answer-model",
        connection_json=None,
        request_status="succeeded",
        request_id=None,
        answer="包含缺失和恶意证据入口。",
        error=None,
        evidence_state="exact",
        evidence=[
            {
                "stable_id": f"frame:{missing_frame_id}",
                "kind": "frame",
                "source": "display:1",
                "start_ms": 120_000,
                "end_ms": 120_000,
                "frame_id": missing_frame_id,
                "resource_path": str(missing_path),
            },
            {
                "stable_id": f"frame:{other_frame_id}",
                "kind": "frame",
                "source": "display:9",
                "start_ms": 240_000,
                "end_ms": 240_000,
                "frame_id": other_frame_id,
                "resource_path": str(other_path),
            },
            {
                "stable_id": f"frame:{outside_frame_id}",
                "kind": "frame",
                "source": "display:1",
                "start_ms": 260_000,
                "end_ms": 260_000,
                "frame_id": outside_frame_id,
                "resource_path": str(outside_path),
            },
            {
                "stable_id": "file:///C:/Windows/win.ini",
                "kind": "frame",
                "source": "display:1",
                "start_ms": 250_000,
                "end_ms": 250_000,
                "frame_id": missing_frame_id,
                "resource_path": "C:/Windows/win.ini",
            },
        ],
    )
    database.finish_session(session_id, "2026-08-04T09:05:00+00:00", "complete")
    database.finish_session(other_session_id, "2026-08-04T08:05:00+00:00", "complete")

    window = MainWindow(
        Settings(data_dir=tmp_path),
        service=JingzhiApplicationService(database, recorder=NoHardwareRecorder()),
    )
    window.show()
    application.processEvents()
    assert window._selected_session_id == session_id

    missing_entry = window.findChild(QPushButton, "answer-evidence-0")
    assert missing_entry is not None
    missing_entry.click()
    application.processEvents()
    assert window._selected_session_id == session_id
    assert "关键帧文件不可读取" in window.evidence_image.text()

    safe_window_start = window._window_start_ms
    cross_session_entry = window.findChild(QPushButton, "answer-evidence-1")
    assert cross_session_entry is not None
    cross_session_entry.click()
    application.processEvents()
    assert window._selected_session_id == session_id
    assert window._window_start_ms == safe_window_start
    assert "不属于当前会话" in window.evidence_image.text()
    assert "display:9" not in window.evidence_metadata.text()

    outside_entry = window.findChild(QPushButton, "answer-evidence-2")
    assert outside_entry is not None
    outside_entry.click()
    application.processEvents()
    assert window._selected_session_id == session_id
    assert window._window_start_ms == safe_window_start
    assert "会话目录以外" in window.evidence_image.text()
    assert "outside.webp" not in window.evidence_metadata.text()

    malicious_entry = window.findChild(QPushButton, "answer-evidence-3")
    assert malicious_entry is not None
    malicious_entry.click()
    application.processEvents()
    assert window._selected_session_id == session_id
    assert window._window_start_ms == safe_window_start
    assert "不受支持" in window.evidence_image.text()
    assert "win.ini" not in window.evidence_metadata.text()
    window.close()


def test_answer_without_evidence_is_explicitly_unconfirmed(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "no-evidence.sqlite3")
    session_id = database.create_session("无证据回答", "2026-08-03T09:00:00+00:00")
    question_id = database.create_question(session_id, 5_000, "发生了什么？", 0, 5_000)
    database.record_answer_version(
        question_id,
        model="answer-model",
        connection_json=None,
        request_status="succeeded",
        request_id=None,
        answer="模型没有按约定标注这段回答。",
        error=None,
        evidence_state="exact",
        evidence=[],
    )
    database.finish_session(session_id, "2026-08-03T09:00:05+00:00", "complete")
    window = MainWindow(
        Settings(data_dir=tmp_path),
        service=JingzhiApplicationService(database, recorder=NoHardwareRecorder()),
    )
    window.show()
    application.processEvents()

    status = window.findChild(QLabel, "answerEvidenceStatus")
    assert status.property("state") == "insufficient"
    assert status.text() == "会话证据不足 · 0 张关键帧 · 0 条字幕"
    assert "## 无法确认" in window.output.raw.toPlainText()
    assert "当前回答没有可核验的会话证据" in window.output.raw.toPlainText()
    window.close()


def test_transcript_detail_supports_diff_undo_and_user_edit(tmp_path: Path, monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "versions.sqlite3")
    session_id = database.create_session("字幕版本", "2026-08-03T09:00:00+00:00")
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 8_000, tmp_path / "audio.wav")
    segment_id = database.add_transcript(
        session_id,
        chunk_id,
        "system",
        1_000,
        3_000,
        "换入便量",
        "zh",
        -0.2,
    )
    database.set_chunk_state(chunk_id, "transcribed")
    database.add_transcript_version(segment_id, "correction", "换入变量", model="correction-small")
    database.configure_transcript_correction(session_id, enabled=True, window_ms=30_000)
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.show()
    application.processEvents()

    transcript_button = window.findChild(QPushButton, f"transcript-{segment_id}")
    assert transcript_button is not None
    transcript_button.click()
    window.transcript_diff_button.click()
    assert "[-便-]{+变+}" in window.evidence_version.text()
    assert window.transcript_undo_button.isVisible()

    window.transcript_undo_button.click()
    assert service.open_session(session_id).transcripts[0].text == "换入便量"

    transcript_button = window.findChild(QPushButton, f"transcript-{segment_id}")
    assert transcript_button is not None
    transcript_button.click()
    monkeypatch.setattr(
        "jingzhi.ui.QInputDialog.getMultiLineText",
        lambda *_args, **_kwargs: ("用户确认：换入变量", True),
    )
    window.transcript_edit_button.click()
    edited = service.open_session(session_id).transcripts[0]
    assert edited.text == "用户确认：换入变量"
    assert edited.version_kind == "user_edit"

    window.close()


def test_dense_timeline_remains_visible_at_required_workspace_sizes(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "dense.sqlite3")
    session_id = database.create_session("高密度时间线", "2026-08-02T09:00:00+00:00")
    image_path = tmp_path / "dense.webp"
    Image.new("RGB", (640, 360), "white").save(image_path)
    cited_frame_id = 0
    for index in range(12):
        frame_id = database.add_frame(
            session_id,
            index * 25_000,
            image_path,
            f"hash-{index}",
            (640, 360),
            source_id=f"display:{index % 2 + 1}",
        )
        if index == 0:
            cited_frame_id = frame_id
    chunk_id = database.add_audio_chunk(session_id, "system", 0, 30_000, tmp_path / "audio.wav")
    transcript_id = database.add_transcript(
        session_id,
        chunk_id,
        "system",
        5_000,
        18_000,
        "高密度时间线中的字幕证据。",
        "zh",
        0.9,
    )
    database.finish_session(session_id, "2026-08-02T09:05:00+00:00", "complete")
    service = VisualStateService(
        database,
        recorder=NoHardwareRecorder(),
        cited_frame_id=cited_frame_id,
        cited_transcript_id=transcript_id,
    )

    for width, height in ((1280, 720), (1600, 900)):
        window = MainWindow(Settings(data_dir=tmp_path), service=service)
        window.resize(width, height)
        window.show()
        application.processEvents()
        assert window.size().width() == width
        assert window.size().height() == height
        assert len(_frame_buttons(window)) == 12
        for name in ("keyframeTrack", "transcriptTrack", "eventTrack"):
            track = window.findChild(QWidget, name)
            assert track.isVisible()
            top_left = track.mapTo(window, track.rect().topLeft())
            bottom_right = track.mapTo(window, track.rect().bottomRight())
            assert 0 <= top_left.x() < width
            assert 0 <= top_left.y() < height
            assert bottom_right.x() < width
            assert bottom_right.y() < height
        assert 190 <= window.findChild(QWidget, "libraryPanel").width() <= 240
        assert 250 <= window.findChild(QWidget, "detailPanel").width() <= 310
        keyframe_scroll = window.findChild(QWidget, "keyframeTrack").findChild(QScrollArea)
        assert keyframe_scroll.horizontalScrollBar().maximum() > 0
        cited_frame = window.findChild(QPushButton, f"keyframe-{cited_frame_id}")
        cited_frame.click()
        application.processEvents()
        assert cited_frame.property("cited") is True
        assert cited_frame.property("selected") is True
        transcript = window.findChild(QPushButton, f"transcript-{transcript_id}")
        transcript.click()
        application.processEvents()
        assert "已校订" in window.evidence_version.text()
        assert _contains_rgb(cited_frame.grab().toImage(), (121, 211, 180))
        assert _contains_rgb(transcript.grab().toImage(), (231, 179, 106))
        assert _contains_rgb(cited_frame.grab().toImage(), (237, 240, 233))
        screenshot = window.grab().toImage()
        assert screenshot.width() == width and screenshot.height() == height
        sampled_colors = {
            screenshot.pixelColor(x, y).rgba()
            for x in range(0, width, max(1, width // 16))
            for y in range(0, height, max(1, height // 12))
        }
        assert len(sampled_colors) > 8
        window.close()


def test_reduced_motion_disables_nonessential_timeline_animations(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("JINGZHI_REDUCE_MOTION", "1")
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "reduced-motion.sqlite3")
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())

    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.show()
    application.processEvents()

    assert window._animations_enabled is False
    assert window._detail_opacity.opacity() == 1.0
    assert EvidenceButton.HOVER_DURATION_MS == 145
    assert window._detail_animation.duration() == 220
    window.close()


def test_quick_question_controls_range_cancel_voice_and_manual_speech(
    tmp_path: Path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "questions.sqlite3")
    database.create_session("即时提问", "2026-08-03T09:00:00+00:00")

    class QuestionRecorder(NoHardwareRecorder):
        is_recording = True

        def __init__(self) -> None:
            self.anchor_calls: list[int] = []
            self.range_calls: list[int] = []
            self.cancel_calls = 0
            self.voice_started = 0
            self.voice_finished = 0
            self.block_voice = False
            self.voice_finish_started = threading.Event()
            self.release_voice_finish = threading.Event()

        def capture_question_anchor(self, lookback_ms: int) -> int:
            self.anchor_calls.append(lookback_ms)
            return 41

        def set_question_range(self, lookback_ms: int) -> None:
            self.range_calls.append(lookback_ms)

        def cancel_question(self) -> bool:
            self.cancel_calls += 1
            return True

        def start_question_voice(self) -> None:
            self.voice_started += 1

        def finish_question_voice(self) -> str:
            self.voice_finished += 1
            if self.block_voice:
                self.voice_finish_started.set()
                self.release_voice_finish.wait(timeout=2)
            return "语音转成的可编辑问题"

    spoken: list[str] = []

    class FakeSpeech:
        def say(self, text: str) -> None:
            spoken.append(text)

    monkeypatch.setattr("jingzhi.ui.QTextToSpeech", FakeSpeech)
    recorder = QuestionRecorder()
    service = JingzhiApplicationService(database, recorder=recorder)
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.show()
    application.processEvents()

    window.capsule_ask_button.click()
    window.capsule_ask_button.click()
    assert recorder.anchor_calls == [2 * 60_000, 2 * 60_000]
    assert window.question.hasFocus()
    assert window.ask_shortcut.context() == Qt.ShortcutContext.ApplicationShortcut
    window.ask_shortcut.activated.emit()
    assert recorder.anchor_calls[-1] == 2 * 60_000

    range_input = window.findChild(QComboBox, "questionRange")
    assert range_input is not None
    assert [range_input.itemData(index) for index in range(range_input.count())] == [
        30_000,
        2 * 60_000,
        5 * 60_000,
    ]
    range_input.setCurrentIndex(0)
    assert recorder.range_calls[-1] == 30_000

    window.voice_button.pressed.emit()
    window.voice_button.released.emit()
    for _ in range(50):
        application.processEvents()
        if window.question.text():
            break
    assert recorder.voice_started == 1
    assert recorder.voice_finished == 1
    assert window.question.text() == "语音转成的可编辑问题"

    window._show_answer(41, "这是默认静音的回答")
    assert spoken == []
    window.speak_button.click()
    assert spoken == ["这是默认静音的回答"]
    window.cancel_question_button.click()
    assert recorder.cancel_calls == 1
    assert window.question.text() == ""

    window.ask_shortcut.activated.emit()
    recorder.block_voice = True
    window.voice_button.pressed.emit()
    window.voice_button.released.emit()
    assert recorder.voice_finish_started.wait(timeout=1)
    window.cancel_question_button.click()
    recorder.release_voice_finish.set()
    for _ in range(50):
        application.processEvents()
    assert recorder.cancel_calls == 2
    assert window.question.text() == ""
    assert window.voice_button.text() == "按住说话"
    window.close()


def test_session_library_search_filter_pin_delete_and_restore(tmp_path, monkeypatch) -> None:
    application = QApplication.instance() or QApplication([])
    now = datetime(2026, 8, 4, 12, tzinfo=UTC)
    database = Database(tmp_path / "library.sqlite3")
    first_id = database.create_session("路线图会议", "2026-08-03T10:00:00+00:00")
    database.finish_session(first_id, "2026-08-03T11:00:00+00:00", "complete")
    second_id = database.create_session("普通会话", "2026-08-02T10:00:00+00:00")
    database.finish_session(second_id, "2026-08-02T11:00:00+00:00", "complete")
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder(), now=lambda: now)
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.show()
    application.processEvents()
    assert window._maintenance_timer.isActive()

    assert window.session_library.count() == 2
    window.session_search.setText("路线图")
    application.processEvents()
    assert window.session_library.count() == 1
    assert first_id == window.session_library.currentItem().data(Qt.ItemDataRole.UserRole)

    window.session_pin_button.click()
    assert "已固定" in window.session_library.currentItem().text()
    monkeypatch.setattr(
        "jingzhi.ui.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    window.session_delete_button.click()
    assert window.session_library.count() == 0

    window.session_search.clear()
    window.session_filter.setCurrentIndex(3)
    application.processEvents()
    assert window.session_library.count() == 1
    assert "回收区" in window.session_library.currentItem().text()
    window.session_restore_button.click()
    application.processEvents()
    assert window.session_filter.currentData() == "all"
    assert window.session_library.count() == 2
    assert {
        window.session_library.item(index).data(Qt.ItemDataRole.UserRole) for index in range(2)
    } == {
        first_id,
        second_id,
    }
    window.close()


def test_material_completion_does_not_cross_selected_session(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "material-ui.sqlite3")
    first_id = database.create_session("第一会话", "2026-08-04T10:00:00+00:00")
    second_id = database.create_session("第二会话", "2026-08-04T11:00:00+00:00")
    material = database.record_material_version(
        first_id,
        kind="generated",
        content="# 第一会话材料",
        template_id=None,
        model="analysis",
        connection_json='{"connection_name":"主连接"}',
        model_invocation_id=None,
        request_status="succeeded",
        request_id="request-1",
        error=None,
        evidence_state="exact",
        evidence=[],
    )
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window._selected_session_id = second_id
    window._material_generation_in_flight = True

    window._show_material(material)

    assert material.id not in window._materials_by_id
    assert window._material_generation_in_flight is False
    assert "材料已保存" in window.status.text()
    window._selected_session_id = first_id
    window._materials_by_id = {material.id: material}
    window._selected_material_version_id = material.id
    window._show_material(material)
    application.processEvents()
    assert window.material_selector.currentData() == material.id
    window.close()


def test_answer_completion_does_not_replace_another_selected_session(tmp_path) -> None:
    window = MainWindow(Settings(data_dir=tmp_path))
    window._selected_session_id = "current-session"

    window._show_answer(99, "过期回答", "original-session")

    assert "原会话" in window.notice_text.text()
    assert window._last_answer == ""
    window.close()


def test_interrupted_session_timeline_retains_status(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "interrupted-ui.sqlite3")
    database.create_session("中断会话", "2026-08-04T10:00:00+00:00")
    service = JingzhiApplicationService(database, recorder=NoHardwareRecorder())
    window = MainWindow(Settings(data_dir=tmp_path), service=service)
    window.show()
    application.processEvents()

    window.session_library.setCurrentItem(window.session_library.item(0))
    application.processEvents()

    assert "已中断" in window.workspace_meta.text()
    window.close()


def test_main_window_reports_audio_recovery_and_retryable_tasks(tmp_path) -> None:
    application = QApplication.instance() or QApplication([])
    database = Database(tmp_path / "jingzhi.sqlite3")
    session_id = database.create_session("失败音频", "2026-01-01T00:00:00+00:00")
    audio = tmp_path / "failed.wav"
    audio.write_bytes(b"audio")
    chunk_id = database.add_audio_chunk(session_id, "microphone", 0, 2_000, audio)
    database.set_chunk_state(chunk_id, "failed", "temporary")
    window = MainWindow(Settings(data_dir=tmp_path))
    window.show()
    application.processEvents()

    window._audio_recovery_finished(SimpleNamespace(queued_chunks=2, missing_chunks=1))

    assert "已恢复 2 个待转写音频片段" in window.notice_text.text()
    assert "音频文件缺失" in window.notice_text.text()
    assert window.retry_audio_button.isVisible()
    window.close()
    application.processEvents()
