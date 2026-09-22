#!/usr/bin/env python3
"""Local video phase annotation. Run with --help for usage."""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile


PHASES = ("approach_start", "contact_active_start", "release_start", "idle_start")
CAMERA = "observation.images.external_view_camera"
MARKER_HIT_TOLERANCE_MS = 30


def format_time(milliseconds: int) -> str:
    seconds, millis = divmod(max(0, round(milliseconds)), 1000)
    return f"{seconds:02d}:{millis:03d}"


def phase_values(record: dict) -> dict[str, float | None]:
    raw = record.get("phase_timestamps") or {}
    if not isinstance(raw, dict):
        raise ValueError("phase_timestamps must be a JSON object.")
    values = {phase: raw.get(phase) for phase in PHASES}
    for phase, value in values.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"Invalid timestamp for {phase}: {value!r}")
    # Zero is the dataset's empty-value marker. Internally, None makes empty
    # phases easy to distinguish, including gaps after marker removal.
    return {phase: None if value in (None, 0) else value for phase, value in values.items()}


class Dataset:
    """Preserve unrelated JSONL lines byte for byte; save under an advisory lock."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.path = self.root / "meta/interaction_metadata.jsonl"
        self.snapshot = self.path.read_bytes()
        self.lines = self.snapshot.splitlines(keepends=True)
        self.records: dict[int, dict] = {}
        self.line_indices: dict[int, int] = {}
        for line_index, line in enumerate(self.lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                episode = record["episode_index"]
                if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
                    raise ValueError("episode_index must be a non-negative integer")
                if episode in self.records:
                    raise ValueError(f"Duplicate episode_index={episode}")
                phase_values(record)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{self.path}, line {line_index + 1}: {exc}") from exc
            self.records[episode] = record
            self.line_indices[episode] = line_index
        if not self.records:
            raise ValueError(f"No episodes found in {self.path}")
        self.videos: dict[int, Path] = {}
        for path in sorted((self.root / "videos").glob(f"chunk-*/{CAMERA}/episode_*.mp4")):
            match = re.fullmatch(r"episode_(\d+)\.mp4", path.name)
            if match and int(match[1]) in self.records:
                episode = int(match[1])
                if episode in self.videos:
                    raise ValueError(f"More than one video was found for episode {episode}")
                self.videos[episode] = path
        if not self.videos:
            raise ValueError(f"No videos found at videos/chunk-*/{CAMERA}/episode_*.mp4")
        self.episodes = sorted(self.records)

    def save(self, episode: int, values: dict[str, float | None]) -> None:
        record = deepcopy(self.records[episode])
        stored_values = {phase: values.get(phase) if values.get(phase) is not None else 0.0
                         for phase in PHASES}
        record["phase_timestamps"] = {
            **(record.get("phase_timestamps") or {}), **stored_values,
        }
        # Remove the progress field written by older versions of this tool.
        record.pop("_phase_annotation_progress", None)
        phase_values(record)  # Reject NaN, infinity and invalid timestamps before writing.
        lines = self.lines.copy()
        index = self.line_indices[episode]
        ending = b"\r\n" if lines[index].endswith(b"\r\n") else b"\n"
        lines[index] = json.dumps(record, ensure_ascii=False, allow_nan=False,
                                  separators=(",", ":")).encode("utf-8") + ending
        content = b"".join(lines)
        with self.path.with_suffix(".jsonl.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.read_bytes() != self.snapshot:
                raise ValueError("Another process changed the file. Restart the annotator "
                                 "to load the changes. The new marker was not saved.")
            backup = self.path.with_suffix(".jsonl.bak")
            try:
                with backup.open("xb") as stream:
                    stream.write(self.snapshot)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=".phases-",
                                                 delete=False) as stream:
                    temporary = Path(stream.name)
                    os.fchmod(stream.fileno(), stat.S_IMODE(self.path.stat().st_mode))
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Also catch edits by non-cooperating writers during backup creation.
                if self.path.read_bytes() != self.snapshot:
                    raise ValueError("Another process changed the file. Restart the annotator.")
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.lines, self.snapshot = lines, content
        self.records[episode] = record


# Keep dataset helpers and --help usable without installing Qt.
def launch(dataset: Dataset, initial_episode: int) -> int:
    try:
        from PySide6.QtCore import QEvent, QPointF, Qt, QTimer, QUrl, Signal
        from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
        from PySide6.QtMultimedia import QMediaPlayer
        from PySide6.QtMultimediaWidgets import QVideoWidget
        from PySide6.QtWidgets import (
            QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
            QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSpinBox,
            QToolTip, QVBoxLayout, QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("Install the dependencies: python -m pip install -r "
                           f"{Path(__file__).with_name('requirements-annotator.txt')}") from exc

    class Timeline(QWidget):
        seek = Signal(int)
        markerSelected = Signal(str)
        markerPreview = Signal(str, int)
        markerDropped = Signal(str, int, int)

        def __init__(self):
            super().__init__()
            self.duration = self.position = 0
            self.values = dict.fromkeys(PHASES)
            self.dragging_phase = None
            self.drag_original = 0
            self.drag_offset_x = 0.0
            self.setFixedHeight(104)
            self.setMouseTracking(True)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setAccessibleName("Video timeline with phase markers")

        def x_at(self, milliseconds):
            return 18 + (self.width() - 36) * milliseconds / max(1, self.duration)

        def markers(self):
            placed = []
            for phase, value in self.values.items():
                if value is None or not self.duration:
                    continue
                x = self.x_at(min(self.duration, round(value * 1000)))
                y = 20
                while any(abs(x - px) < 18 and abs(y - py) < 13 for _, px, py in placed):
                    y += 13
                placed.append((phase, x, y))
            return placed

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("#343c48"), 4, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(18, 70), QPointF(self.width() - 18, 70))
            x = self.x_at(min(self.position, self.duration))
            painter.setPen(QPen(QColor("#e8edf5"), 4))
            painter.drawLine(QPointF(18, 70), QPointF(x, 70))
            for phase, mx, my in self.markers():
                painter.setPen(QPen(QColor("#785136"), 1))
                painter.drawLine(QPointF(mx, my + 7), QPointF(mx, 66))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#ffac62"))
                painter.drawPolygon(QPolygonF([QPointF(mx, my - 6), QPointF(mx + 6, my),
                                               QPointF(mx, my + 6), QPointF(mx - 6, my)]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QPointF(x, 70), 6, 6)
            painter.setPen(QColor("#919cab"))
            painter.drawText(18, 97, "00:000")
            label = format_time(self.duration)
            painter.drawText(self.width() - 18 - painter.fontMetrics().horizontalAdvance(label),
                             97, label)

        def marker_at(self, point):
            return next((phase for phase, x, y in self.markers()
                         if abs(point.x() - x) <= 9 and abs(point.y() - y) <= 9), None)

        def milliseconds_at(self, x):
            fraction = (x - 18) / max(1, self.width() - 36)
            return round(max(0, min(1, fraction)) * self.duration)

        def constrained_milliseconds(self, phase, milliseconds):
            index = PHASES.index(phase)
            earlier = [self.values[p] for p in PHASES[:index] if self.values[p] is not None]
            later = [self.values[p] for p in PHASES[index + 1:] if self.values[p] is not None]
            minimum = round(max(earlier) * 1000) if earlier else 0
            maximum = round(min(later) * 1000) if later else self.duration
            return max(minimum, min(maximum, milliseconds))

        def seek_at(self, event):
            if not self.duration:
                return
            phase = self.marker_at(event.position())
            if phase:
                self.seek.emit(round(self.values[phase] * 1000))
            else:
                self.seek.emit(self.milliseconds_at(event.position().x()))

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                phase = self.marker_at(event.position())
                if phase:
                    self.dragging_phase = phase
                    self.drag_original = round(self.values[phase] * 1000)
                    marker_x = next(x for p, x, _ in self.markers() if p == phase)
                    self.drag_offset_x = event.position().x() - marker_x
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self.seek.emit(self.drag_original)
                    self.markerSelected.emit(phase)
                else:
                    self.seek_at(event)

        def mouseMoveEvent(self, event):
            if self.dragging_phase and event.buttons() & Qt.MouseButton.LeftButton:
                milliseconds = self.milliseconds_at(event.position().x() - self.drag_offset_x)
                milliseconds = self.constrained_milliseconds(self.dragging_phase, milliseconds)
                self.values[self.dragging_phase] = round(milliseconds / 1000, 3)
                self.markerPreview.emit(self.dragging_phase, milliseconds)
                QToolTip.showText(event.globalPosition().toPoint(),
                                  f"{self.dragging_phase} · {format_time(milliseconds)}", self)
                self.update()
                return
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.seek_at(event)
            phase = self.marker_at(event.position())
            if phase:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                QToolTip.showText(event.globalPosition().toPoint(),
                                  f"{phase} · {format_time(self.values[phase] * 1000)} "
                                  "· Space removes", self)
            else:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                QToolTip.hideText()

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton and self.dragging_phase:
                phase = self.dragging_phase
                milliseconds = round(self.values[phase] * 1000)
                original = self.drag_original
                self.dragging_phase = None
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                QToolTip.hideText()
                if milliseconds != original:
                    self.markerDropped.emit(phase, milliseconds, original)

        def leaveEvent(self, event):
            if not self.dragging_phase:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                QToolTip.hideText()

        def cancel_drag(self):
            self.dragging_phase = None
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            QToolTip.hideText()

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.episode = initial_episode
            self.values = dict.fromkeys(PHASES)
            self.selected_marker_phase = None
            self.ready = False
            self.setWindowTitle("Phase Studio · Interaction annotation")
            self.resize(1220, 880)
            self.setMinimumSize(960, 720)
            self.player = QMediaPlayer(self)
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(28, 22, 28, 18)
            layout.setSpacing(14)
            header = QHBoxLayout()
            brand = QVBoxLayout()
            title = QLabel("PHASE STUDIO")
            title.setObjectName("brand")
            brand.addWidget(title)
            subtitle = QLabel(dataset.root.name)
            subtitle.setObjectName("muted")
            subtitle.setToolTip(str(dataset.root))
            brand.addWidget(subtitle)
            header.addLayout(brand)
            header.addStretch()
            header.addWidget(QLabel("Episode ID"))
            self.episode_input = QSpinBox()
            self.episode_input.setRange(0, max(dataset.episodes))
            self.episode_input.setValue(initial_episode)
            self.episode_input.setKeyboardTracking(False)
            self.episode_input.setFixedWidth(112)
            self.episode_input.setToolTip("Enter an ID and press Enter")
            self.episode_input.editingFinished.connect(self.open_entered_episode)
            header.addWidget(self.episode_input)
            self.previous = self.button("←", lambda: self.neighbor(-1))
            self.next = self.button("→", lambda: self.neighbor(1))
            header.addWidget(self.previous)
            header.addWidget(self.next)
            layout.addLayout(header)

            self.details = QLabel()
            self.details.setObjectName("muted")
            layout.addWidget(self.details)
            video_frame = QFrame()
            video_frame.setObjectName("videoFrame")
            video_layout = QVBoxLayout(video_frame)
            video_layout.setContentsMargins(1, 1, 1, 1)
            self.video = QVideoWidget()
            self.video.setMinimumHeight(240)
            self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            video_layout.addWidget(self.video)
            self.player.setVideoOutput(self.video)
            layout.addWidget(video_frame, 1)

            controls = QHBoxLayout()
            self.play = self.button("▶  Play", self.toggle_play)
            self.play.setMinimumWidth(185)
            controls.addWidget(self.play)
            self.time = QLabel("00:000  /  00:000")
            self.time.setObjectName("time")
            controls.addWidget(self.time)
            controls.addStretch()
            controls.addWidget(QLabel("Speed"))
            self.speed = QComboBox()
            for rate in (0.25, 0.5, 0.75, 1, 1.5, 2):
                self.speed.addItem(f"{rate}×", rate)
            self.speed.setCurrentIndex(3)
            self.speed.currentIndexChanged.connect(
                lambda: self.player.setPlaybackRate(self.speed.currentData())
            )
            controls.addWidget(self.speed)
            layout.addLayout(controls)
            self.timeline = Timeline()
            self.timeline.seek.connect(self.seek)
            self.timeline.markerSelected.connect(self.select_marker)
            self.timeline.markerPreview.connect(self.preview_marker)
            self.timeline.markerDropped.connect(self.drop_marker)
            layout.addWidget(self.timeline)

            phases_layout = QHBoxLayout()
            self.phase_buttons = {}
            for index, phase in enumerate(PHASES):
                button = self.button("", lambda checked=False, p=phase: self.seek_phase(p))
                button.setMinimumHeight(76)
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                button.setToolTip(
                    f"{phase}: click to move here, then press Space to remove the marker"
                )
                button.setAccessibleName(phase)
                phases_layout.addWidget(button)
                self.phase_buttons[phase] = button
            layout.addLayout(phases_layout)

            actions = QHBoxLayout()
            self.mark_button = self.button("Space · Add marker", self.mark)
            self.mark_button.setObjectName("primary")
            self.mark_button.setMinimumWidth(320)
            actions.addWidget(self.mark_button)
            self.undo_button = self.button("Undo last marker", self.undo)
            actions.addWidget(self.undo_button)
            actions.addStretch()
            actions.addWidget(self.button("Reset phases", self.reset))
            layout.addLayout(actions)
            self.status = QLabel()
            self.status.setObjectName("status")
            self.status.setMinimumHeight(28)
            layout.addWidget(self.status)
            help_label = QLabel("SPACE  add / remove marker     ENTER  next episode     R / K  pause / play     "
                                "← / →  speed     Ctrl+Z  undo     Alt+← / →  episode")
            help_label.setObjectName("muted")
            layout.addWidget(help_label)

            self.player.positionChanged.connect(self.update_time)
            self.player.durationChanged.connect(self.update_time)
            self.player.playbackStateChanged.connect(self.playback_changed)
            self.player.mediaStatusChanged.connect(self.media_status)
            self.player.errorOccurred.connect(self.media_error)
            self.shortcuts = []
            for key, callback in (("K", self.toggle_play), ("R", self.toggle_play),
                                  ("Ctrl+Z", self.undo),
                                  ("Ctrl+C", self.close),
                                  ("Alt+Left", lambda: self.neighbor(-1)),
                                  ("Alt+Right", lambda: self.neighbor(1))):
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.setAutoRepeat(False)
                shortcut.activated.connect(callback)
                self.shortcuts.append(shortcut)
            QApplication.instance().installEventFilter(self)
            self.timer = QTimer(self)
            self.timer.setInterval(20)
            self.timer.timeout.connect(self.update_time)
            self.timer.start()
            self.load_episode(initial_episode)

        @staticmethod
        def button(text, callback):
            button = QPushButton(text)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(callback)
            return button

        def eventFilter(self, watched, event):
            if (self.isActiveWindow() and QApplication.activeModalWidget() is None
                    and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)):
                focus = QApplication.focusWidget()
                editing = isinstance(focus, (QLineEdit, QSpinBox, QComboBox))
                if (event.modifiers() == Qt.KeyboardModifier.NoModifier
                        and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
                    if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                        if focus is self.episode_input or self.episode_input.isAncestorOf(focus):
                            self.episode_input.interpretText()
                            self.open_entered_episode()
                        else:
                            self.neighbor(1)
                    return True
                if not editing and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                    if event.key() == Qt.Key.Key_Space:
                        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                            self.mark()
                        return True
                    if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                        if event.type() == QEvent.Type.KeyPress:
                            self.adjust_speed(-1 if event.key() == Qt.Key.Key_Left else 1)
                        return True
            return super().eventFilter(watched, event)

        def open_entered_episode(self):
            episode = self.episode_input.value()
            if episode != self.episode:
                self.load_episode(episode)
            self.episode_input.clearFocus()
            self.video.setFocus()

        def neighbor(self, direction):
            index = dataset.episodes.index(self.episode) + direction
            if 0 <= index < len(dataset.episodes):
                self.load_episode(dataset.episodes[index])

        def load_episode(self, episode):
            if episode not in dataset.records:
                self.status.setText(f"Episode {episode} is not in interaction_metadata.jsonl.")
                self.episode_input.setValue(self.episode)
                return
            self.player.stop()
            self.timeline.cancel_drag()
            self.selected_marker_phase = None
            self.ready = False
            self.episode = episode
            self.values = phase_values(dataset.records[episode])
            self.episode_input.setValue(episode)
            record = dataset.records[episode]
            index = dataset.episodes.index(episode)
            self.previous.setEnabled(index > 0)
            self.next.setEnabled(index < len(dataset.episodes) - 1)
            self.details.setText(f"{index + 1} / {len(dataset.episodes)}   ·   "
                                 f"{record.get('interaction_class', '—')}   ·   "
                                 f"{record.get('person', '—')}   ·   external_view_camera")
            self.status.setText("Annotation loaded · changes are saved automatically")
            self.player.setSource(QUrl())
            self.timeline.duration = self.timeline.position = 0
            self.refresh()
            self.update_time()
            path = dataset.videos.get(episode)
            if path is None or not path.is_file():
                self.status.setText(f"No video found for episode {episode} in {CAMERA}.")
                return
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            # Start decoding the first frame, then remain paused for annotation.
            self.player.pause()

        def media_status(self, status):
            self.ready = status in (QMediaPlayer.MediaStatus.LoadedMedia,
                                    QMediaPlayer.MediaStatus.BufferedMedia,
                                    QMediaPlayer.MediaStatus.BufferingMedia,
                                    QMediaPlayer.MediaStatus.EndOfMedia)
            self.mark_button.setEnabled(self.ready)
            self.play.setEnabled(self.ready)

        def media_error(self, error, message):
            self.ready = False
            self.mark_button.setEnabled(False)
            self.play.setEnabled(False)
            self.status.setText(f"Video error: {message}")
            self.status.setToolTip(str(dataset.videos.get(self.episode, "")))

        def update_time(self, *args):
            self.timeline.position = self.player.position()
            self.timeline.duration = self.player.duration()
            self.time.setText(f"{format_time(self.player.position())}  /  "
                              f"{format_time(self.player.duration())}")
            self.timeline.update()

        def playback_changed(self, state):
            self.play.setText("Ⅱ  Pause" if state == QMediaPlayer.PlaybackState.PlayingState
                              else "▶  Play")

        def toggle_play(self):
            if not self.ready:
                return
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                if self.player.position() >= self.player.duration():
                    self.player.setPosition(0)
                self.player.play()

        def adjust_speed(self, direction):
            new_index = max(0, min(self.speed.count() - 1,
                                   self.speed.currentIndex() + direction))
            self.speed.setCurrentIndex(new_index)
            self.status.setText(f"Playback speed · {self.speed.currentText()}")

        def seek(self, milliseconds):
            if self.ready:
                self.player.pause()
                self.player.setPosition(max(0, min(self.player.duration(), milliseconds)))
                self.update_time()

        def seek_phase(self, phase):
            if self.values[phase] is not None:
                self.seek(round(self.values[phase] * 1000))
                self.selected_marker_phase = phase

        def select_marker(self, phase):
            self.selected_marker_phase = phase

        def preview_marker(self, phase, milliseconds):
            self.values[phase] = round(milliseconds / 1000, 3)
            self.refresh()
            self.seek(milliseconds)
            self.status.setText(f"↔ {phase} · {format_time(milliseconds)} · release to save")

        def drop_marker(self, phase, milliseconds, original):
            if milliseconds == 0:
                self.remove_marker(phase)
                return
            values = {**self.values, phase: round(milliseconds / 1000, 3)}
            if not self.persist(values):
                self.values[phase] = round(original / 1000, 3)
                self.refresh()
                self.seek(original)

        def refresh(self):
            next_phase = next((p for p in PHASES if self.values[p] is None), None)
            for index, (phase, button) in enumerate(self.phase_buttons.items()):
                value = self.values[phase]
                button.setText(f"{index + 1:02d}   {phase}\n"
                               + (format_time(value * 1000) if value is not None else "— — : — — —"))
                button.setProperty("phaseState", "next" if phase == next_phase
                                   else "set" if value is not None else "empty")
                button.style().unpolish(button)
                button.style().polish(button)
            self.timeline.values = self.values.copy()
            self.timeline.update()
            self.mark_button.setText(f"Space · {next_phase}" if next_phase else "✓ All 4 phases marked")
            self.mark_button.setEnabled(self.ready)
            self.play.setEnabled(self.ready)
            self.undo_button.setEnabled(any(v is not None for v in self.values.values()))

        def marker_at_current_time(self):
            position = self.player.position()
            selected = self.selected_marker_phase
            if (selected and self.values.get(selected) is not None
                    and abs(round(self.values[selected] * 1000) - position)
                    <= MARKER_HIT_TOLERANCE_MS):
                return selected
            candidates = [
                (abs(round(value * 1000) - position), -index, phase)
                for index, (phase, value) in enumerate(self.values.items())
                if value is not None
                and abs(round(value * 1000) - position) <= MARKER_HIT_TOLERANCE_MS
            ]
            return min(candidates)[2] if candidates else None

        def remove_marker(self, phase):
            self.player.pause()
            if self.persist({**self.values, phase: None}):
                self.selected_marker_phase = None
                self.status.setText(f"✓ Removed {phase} · saved as 0.0")

        def persist(self, values):
            try:
                dataset.save(self.episode, values)
            except (OSError, ValueError) as exc:
                self.player.pause()
                self.status.setText("Save error · the new marker was not applied")
                QMessageBox.critical(self, "Could not save", str(exc))
                return False
            self.values = values
            self.refresh()
            self.status.setText(f"✓ Saved · episode {self.episode} · meta/interaction_metadata.jsonl")
            return True

        def complete_alert(self):
            self.player.pause()
            self.status.setText("✓ All timestamps are marked and saved. "
                                "You can move to the next episode.")
            QApplication.beep()

        def mark(self):
            if not self.ready or self.player.duration() <= 0:
                self.status.setText("Wait for the video to load before adding markers.")
                return
            marker = self.marker_at_current_time()
            if marker is not None:
                self.remove_marker(marker)
                return
            phase = next((p for p in PHASES if self.values[p] is None), None)
            if phase is None:
                self.complete_alert()
                return
            timestamp = round(self.player.position() / 1000, 3)
            if timestamp == 0:
                self.status.setText("00:000 is reserved for an empty timestamp. Move forward first.")
                return
            index = PHASES.index(phase)
            earlier = [self.values[p] for p in PHASES[:index] if self.values[p] is not None]
            later = [self.values[p] for p in PHASES[index + 1:] if self.values[p] is not None]
            if (earlier and timestamp < max(earlier)) or (later and timestamp > min(later)):
                self.status.setText("Phases must follow this order: approach → contact → release → idle.")
                return
            values = {**self.values, phase: timestamp}
            if self.persist(values) and all(v is not None for v in values.values()):
                self.complete_alert()

        def undo(self):
            phase = next((p for p in reversed(PHASES) if self.values[p] is not None), None)
            if phase is not None:
                self.player.pause()
                self.persist({**self.values, phase: None})

        def reset(self):
            if not any(v is not None for v in self.values.values()):
                return
            self.player.pause()
            reply = QMessageBox.question(self, "Reset phases?",
                                         f"Remove all four markers from episode {self.episode}?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if self.persist(dict.fromkeys(PHASES)):
                    self.seek(0)

        def closeEvent(self, event):
            self.timer.stop()
            QApplication.instance().removeEventFilter(self)
            self.player.stop()
            self.player.setSource(QUrl())
            super().closeEvent(event)

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setFont(QFont("DejaVu Sans", 10))
    app.setStyleSheet("""
        QWidget { background: #11161e; color: #e5eaf2; }
        QLabel { background: transparent; }
        QLabel#brand { font-size: 22px; font-weight: 700; letter-spacing: 3px; }
        QLabel#muted { color: #8995a7; font-size: 11px; }
        QLabel#time { font-family: monospace; font-size: 19px; padding-left: 12px; }
        QLabel#status { color: #ffba7c; font-size: 12px; }
        QFrame#videoFrame { background: #080b10; border: 1px solid #2c3542; border-radius: 10px; }
        QPushButton, QSpinBox, QComboBox {
            background: #1c2430; border: 1px solid #323d4e; border-radius: 7px;
            padding: 10px 12px; font-size: 12px;
        }
        QPushButton:hover { background: #293444; border-color: #6c7d93; }
        QPushButton:focus { border-color: #ffac62; }
        QPushButton:disabled { color: #586476; border-color: #26303d; }
        QPushButton#primary { background: #ffac62; color: #1c1b1b; font-weight: 600; }
        QPushButton#primary:hover { background: #ffc18b; }
        QPushButton#primary:disabled { background: #493729; color: #a08a79; }
        QPushButton[phaseState="next"] { border-color: #ffac62; background: #2b251f; }
        QPushButton[phaseState="set"] { color: #ffba7c; }
        QPushButton[phaseState="empty"] { color: #8490a2; }
        QToolTip { background: #263141; color: #ffd1a8; border: 1px solid #ffac62; padding: 6px; }
        QMessageBox QPushButton { min-width: 80px; }
    """)
    window = Window()
    window.show()
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: window.close())
    try:
        return app.exec()
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate interaction phases in local videos.")
    parser.add_argument("dataset", type=Path, help="Dataset root with videos/ and meta/ folders")
    parser.add_argument("--episode", type=int, help="Episode ID to open at startup (default: first episode with a video)")
    args = parser.parse_args()
    try:
        dataset = Dataset(args.dataset)
        episode = min(dataset.videos) if args.episode is None else args.episode
        if episode not in dataset.records:
            raise ValueError(f"Episode {episode} is not in interaction_metadata.jsonl")
        return launch(dataset, episode)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
