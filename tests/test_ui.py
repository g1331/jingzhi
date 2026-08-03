import os
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    QWidget,
)

from jingzhi.application import JingzhiApplicationService
from jingzhi.config import Settings
from jingzhi.database import Database
from jingzhi.ui import EvidenceButton, MainWindow


class NoHardwareRecorder:
    is_recording = False

    def start(self, title, **_kwargs):
        raise AssertionError(f"Unexpected hardware start for {title}")

    def stop(self):
        return None


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
