"""Real-time serial data visualization — Fixed-World 3D Topographic Mapper

v3 — PRD Implementation: Fixed-Reference OpenGL Point Cloud
  FR-1: Fixed world coordinate system — axes never move automatically.
  FR-2: Sensor locked at origin (0, 0, 0) at all times.
  FR-3: True 3D spatial rendering — Z = distance measurement.
  FR-4: Stable camera — user pan/rotate/zoom persists across redraws.
  FR-5: Incremental rendering — points appended, no full rebuild.
  FR-6: Permanent reference grid at fixed world coordinates.
  FR-7: Fixed bounding volume (Small / Medium / Large room presets).
  FR-8: Point cloud persistence — all points remain forever.
  FR-9: Scan path trajectory line.
  FR-10: Distance colour mapping (blue→green→yellow→red).
  NFR-2: PyQtGraph OpenGL for hardware-accelerated rendering.
  NFR-3: float64 storage throughout.
"""

import sys
import random
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import serial
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPlainTextEdit, QPushButton, QSlider, QComboBox, QCheckBox,
    QGroupBox, QScrollArea,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor, QColor
from PyQt5.QtGui import QVector3D  # add this at the top with other PyQt5 imports

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PORT      = "/dev/cu.usbmodem111427801"
BAUD      = 9600
MAX_LOG_LINES = 1000

ANGLE_X_MIN = 40.0
ANGLE_X_MAX = 180.0
ANGLE_Y_MIN = 0.0
ANGLE_Y_MAX = 30.0

REDRAW_INTERVAL_MS = 100
MAX_POINT_STORE    = 50_000

# Room presets (FR-7)
ROOM_PRESETS = {
    "Small  (±300 cm)":  300,
    "Medium (±500 cm)":  500,
    "Large  (±1000 cm)": 1000,
}
DEFAULT_ROOM = "Medium (±500 cm)"

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "object_counter.pkl"
OBJECT_COUNTER_TRAIN_SAMPLES = 240
GRID_PAN_POINTS = 35
GRID_TILT_POINTS = 6

# Angular resolution for the previous-distance lookup map (matches Arduino's
# CELL_DEG = 1.0).  Angles are rounded to this many decimal places before
# being used as dict keys so that readings within the same physical cell
# always map to the same entry.
PREV_DIST_CELL_DEG = 1

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
BG_COLOR      = "#101d3b"
CARD_COLOR    = "#14274f"
ACCENT_COLOR  = "#375fbe"
TEXT_COLOR    = "#d7e3ff"
SUCCESS_COLOR = "#5c88cc"
WARNING_COLOR = "#4f79be"
MUTED_COLOR   = "#7f98c8"

DELTA_POS_COLOR = "#5cf07a"   # green  — object moved closer
DELTA_NEG_COLOR = "#f0a05c"   # orange — object moved away
DELTA_NEU_COLOR = "#7f98c8"   # muted  — no change

# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PAN_CENTRE = (ANGLE_X_MIN + ANGLE_X_MAX) / 2.0  # 110°


def angles_to_xyz(pan_deg, tilt_deg, dist_cm):
    """Spherical → Cartesian.  All inputs may be scalars or arrays."""
    pan  = np.deg2rad(np.asarray(pan_deg,  dtype=np.float64) - PAN_CENTRE)
    tilt = np.deg2rad(np.asarray(tilt_deg, dtype=np.float64))
    dist = np.asarray(dist_cm, dtype=np.float64)
    r_xy = dist * np.cos(tilt)
    x    =  r_xy * np.cos(pan)
    y    =  r_xy * np.sin(pan)
    z    =  dist * np.sin(tilt)
    return x, y, z


def dist_to_rgba(dists, max_dist):
    """
    FR-10: Map distances to RGBA colours (blue→green→yellow→red).
    Returns float32 array of shape (N, 4) with values in [0, 1].
    """
    t = np.clip(np.asarray(dists, dtype=np.float32) / max(float(max_dist), 1.0), 0.0, 1.0)
    r = np.where(t < 0.5, 0.0,        np.where(t < 0.75, (t - 0.5) * 4.0, 1.0)).astype(np.float32)
    g = np.where(t < 0.25, t * 4.0,   np.where(t < 0.75, 1.0, 1.0 - (t - 0.75) * 4.0)).astype(np.float32)
    b = np.where(t < 0.25, 1.0,       np.where(t < 0.5,  1.0 - (t - 0.25) * 4.0, 0.0)).astype(np.float32)
    a = np.ones_like(r)
    return np.column_stack([r, g, b, a])


def _angle_key(pan_deg: float, tilt_deg: float) -> tuple:
    """Quantize (pan, tilt) to the nearest PREV_DIST_CELL_DEG for map lookup."""
    factor = 1.0 / PREV_DIST_CELL_DEG
    return (round(pan_deg * factor) / factor,
            round(tilt_deg * factor) / factor)


# ═══════════════════════════════════════════════════════════════════════════════
# POINT STORE  (NFR-3: float64, FR-8: persistent)
# ═══════════════════════════════════════════════════════════════════════════════
class PointStore:
    def __init__(self, capacity: int = MAX_POINT_STORE):
        self.cap   = capacity
        self._pan  = np.empty(capacity, dtype=np.float64)
        self._tilt = np.empty(capacity, dtype=np.float64)
        self._dist = np.empty(capacity, dtype=np.float64)
        self._x    = np.empty(capacity, dtype=np.float64)
        self._y    = np.empty(capacity, dtype=np.float64)
        self._z    = np.empty(capacity, dtype=np.float64)
        self.count = 0
        self._new_since_render = 0   # how many points appended since last render
        # circular buffer write pointer
        self._next = 0
        # when True the buffer has wrapped at least once
        self._full = False
        # map quantized (pan, tilt) -> index in arrays for quick overwrite
        self._index_map = {}

    def add(self, pan: float, tilt: float, dist: float):
        # quantize angles to avoid tiny floating point differences
        key = (round(float(pan), 3), round(float(tilt), 3))

        # if we already have a point at this (pan, tilt), overwrite it
        if key in self._index_map:
            idx = self._index_map[key]
        else:
            idx = self._next
            # if overwriting an old slot with a different key, remove old mapping
            if self._full:
                old_key = (round(float(self._pan[idx]), 3), round(float(self._tilt[idx]), 3))
                if old_key in self._index_map and self._index_map[old_key] == idx:
                    del self._index_map[old_key]

            self._index_map[key] = idx
            # advance circular pointer
            self._next = (self._next + 1) % self.cap
            if not self._full and self._next == 0:
                self._full = True
            if not self._full:
                # increasing count until full
                self.count += 1

        x, y, z = angles_to_xyz(pan, tilt, dist)
        self._pan[idx]  = pan
        self._tilt[idx] = tilt
        self._dist[idx] = dist
        self._x[idx]    = float(x)
        self._y[idx]    = float(y)
        self._z[idx]    = float(z)
        self._new_since_render += 1

    def arrays(self):
        n = self.count
        if n == 0:
            return (np.array([], dtype=self._pan.dtype),) * 6

        if not self._full:
            return (self._pan[:n], self._tilt[:n], self._dist[:n],
                    self._x[:n],   self._y[:n],    self._z[:n])

        # when full, return arrays in chronological order from oldest to newest
        idxs = np.concatenate((np.arange(self._next, self.cap), np.arange(0, self._next)))
        return (self._pan[idxs], self._tilt[idxs], self._dist[idxs],
                self._x[idxs],   self._y[idxs],    self._z[idxs])

    def clear(self):
        self.count = 0
        self._new_since_render = 0
        self._next = 0
        self._full = False
        self._index_map.clear()

    def consume_new_flag(self):
        had = self._new_since_render
        self._new_since_render = 0
        return had > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SERIAL READER THREAD
# ═══════════════════════════════════════════════════════════════════════════════
class SerialReaderThread(QThread):
    line_received     = pyqtSignal(str)
    connection_failed = pyqtSignal(str)
    connected         = pyqtSignal()
    disconnected      = pyqtSignal()

    def __init__(self, port: str, baud: int):
        super().__init__()
        self.port    = port
        self.baud    = baud
        self.ser     = None
        self.running = False

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.connected.emit()
            self.running = True
            while self.running:
                try:
                    if self.ser.in_waiting:
                        raw  = self.ser.readline()
                        line = raw.decode("utf-8", errors="replace").strip()
                        if line:
                            self.line_received.emit(line)
                except Exception as e:
                    self.connection_failed.emit(str(e))
                    break
        except serial.SerialException as e:
            self.connection_failed.emit(str(e))

    def stop(self):
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.disconnected.emit()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Room Mapper v3 — Fixed-World OpenGL Point Cloud")
        self.setGeometry(60, 40, 1700, 1020)

        self.serial_thread   = None
        self.store           = PointStore()
        self.readings_buffer = deque(maxlen=500)

        self.last_valid_xyz  = None
        self.total_readings  = 0
        self.line_history    = deque(maxlen=MAX_LOG_LINES)

        self._point_size     = 4.0
        self._room_half      = ROOM_PRESETS[DEFAULT_ROOM]
        self._show_path      = True
        self._show_grid      = True
        self._object_counter = None
        self._last_scan_features = None
        self._auto_pan_enabled = False
        self._last_reset_press = 0.0
        self._auto_pan_press_threshold = 1.2
        self._auto_pan_azimuth = -60.0

        # ── Previous-distance map ──────────────────────────────────────────
        # Maps quantized (pan_deg, tilt_deg) → last stored distance (cm).
        # Mirrors the Arduino's cell_dist[][] grid so the UI can display
        # "previous reading at this angle" and the delta from the current one.
        self._prev_dist_map: dict[tuple, float] = {}

        # GL items (created once, updated incrementally — FR-5)
        self._scatter_item   = None
        self._path_item      = None
        self._origin_item    = None

        self._setup_ui()
        self._apply_styling()
        self._build_gl_scene()
        self._start_serial_connection()
        self._load_or_build_object_counter()

        self._auto_pan_timer = QTimer(self)
        self._auto_pan_timer.setInterval(80)
        self._auto_pan_timer.timeout.connect(self._advance_auto_pan)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(REDRAW_INTERVAL_MS)
        self._redraw_timer.timeout.connect(self._maybe_redraw)
        self._redraw_timer.start()

    # ── UI construction ──────────────────────────────────────────────────────
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        central.setLayout(root)

        # ── Left panel ───────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(300)
        ll = QVBoxLayout()
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        left.setLayout(ll)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)
        left_scroll.setFixedWidth(320)

        title = QLabel("TOPO MAPPER v3")
        title.setFont(QFont("Monaco", 16, QFont.Bold))
        title.setStyleSheet(f"color: {SUCCESS_COLOR}; padding: 6px 0;")
        ll.addWidget(title)

        self.status_label = QLabel("● DISCONNECTED")
        self.status_label.setFont(QFont("Monaco", 9, QFont.Bold))
        self.status_label.setStyleSheet(f"color: {WARNING_COLOR}; padding: 8px;")
        ll.addWidget(self._card(self.status_label, "Connection"))

        self.total_label = QLabel("0")
        self.total_label.setFont(QFont("Monaco", 13, QFont.Bold))
        ll.addWidget(self._card(self.total_label, "Points Stored"))

        self.object_count_label = QLabel("—")
        self.object_count_label.setFont(QFont("Monaco", 13, QFont.Bold))
        self.object_count_label.setStyleSheet(f"color: {ACCENT_COLOR};")
        ll.addWidget(self._card(self.object_count_label, "Estimated Objects"))

        self.occupancy_label = QLabel("—")
        self.occupancy_label.setFont(QFont("Monaco", 11, QFont.Bold))
        self.occupancy_label.setStyleSheet(f"color: {SUCCESS_COLOR};")
        ll.addWidget(self._card(self.occupancy_label, "Room Occupancy"))

        self.stability_label = QLabel("—")
        self.stability_label.setFont(QFont("Monaco", 10))
        ll.addWidget(self._card(self.stability_label, "Scene Stability"))

        self.avg_dist_label = QLabel("—")
        self.avg_dist_label.setFont(QFont("Monaco", 10))
        ll.addWidget(self._card(self.avg_dist_label, "Avg Distance"))

        # ── Distance readout card: current / previous / delta ────────────────
        dist_widget = QWidget()
        dist_widget.setStyleSheet("border: none;")
        dist_layout = QVBoxLayout()
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(2)
        dist_widget.setLayout(dist_layout)

        self.dist_label = QLabel("—")
        self.dist_label.setFont(QFont("Monaco", 12, QFont.Bold))
        self.dist_label.setStyleSheet(f"color: {SUCCESS_COLOR}; border: none;")
        dist_layout.addWidget(self.dist_label)

        prev_row = QWidget()
        prev_row.setStyleSheet("border: none;")
        prev_row_layout = QHBoxLayout()
        prev_row_layout.setContentsMargins(0, 0, 0, 0)
        prev_row_layout.setSpacing(6)
        prev_row.setLayout(prev_row_layout)

        prev_lbl_head = QLabel("prev:")
        prev_lbl_head.setFont(QFont("Monaco", 8))
        prev_lbl_head.setStyleSheet(f"color: {MUTED_COLOR}; border: none;")
        prev_row_layout.addWidget(prev_lbl_head)

        self.prev_dist_label = QLabel("—")
        self.prev_dist_label.setFont(QFont("Monaco", 8, QFont.Bold))
        self.prev_dist_label.setStyleSheet(f"color: {TEXT_COLOR}; border: none;")
        prev_row_layout.addWidget(self.prev_dist_label)

        prev_row_layout.addStretch()
        dist_layout.addWidget(prev_row)

        delta_row = QWidget()
        delta_row.setStyleSheet("border: none;")
        delta_row_layout = QHBoxLayout()
        delta_row_layout.setContentsMargins(0, 0, 0, 0)
        delta_row_layout.setSpacing(6)
        delta_row.setLayout(delta_row_layout)

        delta_lbl_head = QLabel("Δ:")
        delta_lbl_head.setFont(QFont("Monaco", 8))
        delta_lbl_head.setStyleSheet(f"color: {MUTED_COLOR}; border: none;")
        delta_row_layout.addWidget(delta_lbl_head)

        self.delta_dist_label = QLabel("—")
        self.delta_dist_label.setFont(QFont("Monaco", 9, QFont.Bold))
        self.delta_dist_label.setStyleSheet(f"color: {DELTA_NEU_COLOR}; border: none;")
        delta_row_layout.addWidget(self.delta_dist_label)

        delta_row_layout.addStretch()
        dist_layout.addWidget(delta_row)

        ll.addWidget(self._card(dist_widget, "Last Distance  ·  Prev @ Angle  ·  Δ"))

        self.pan_label = QLabel("—")
        self.pan_label.setFont(QFont("Monaco", 10))
        ll.addWidget(self._card(self.pan_label, "Pan (X)"))

        self.tilt_label = QLabel("—")
        self.tilt_label.setFont(QFont("Monaco", 10))
        ll.addWidget(self._card(self.tilt_label, "Tilt (Y)"))

        self.raw_label = QLabel("—")
        self.raw_label.setFont(QFont("Monaco", 7))
        self.raw_label.setWordWrap(True)
        ll.addWidget(self._card(self.raw_label, "Last Raw Line"))

        # Room preset (FR-7)
        room_box = QComboBox()
        room_box.setFont(QFont("Monaco", 8))
        for name in ROOM_PRESETS:
            room_box.addItem(name)
        room_box.setCurrentText(DEFAULT_ROOM)
        room_box.currentTextChanged.connect(self._on_room_changed)
        ll.addWidget(self._card(room_box, "Room Size Preset"))

        # Point size
        self._ps_label = QLabel(f"Point size: {self._point_size:.0f}")
        self._ps_label.setFont(QFont("Monaco", 8))
        ps_slider = QSlider(Qt.Horizontal)
        ps_slider.setMinimum(1); ps_slider.setMaximum(20)
        ps_slider.setValue(int(self._point_size))
        ps_slider.valueChanged.connect(self._on_point_size_changed)
        ll.addWidget(self._card(self._ps_label, "Point Size"))
        ll.addWidget(ps_slider)

        # Toggles
        self._path_chk = QCheckBox("Show scan path")
        self._path_chk.setFont(QFont("Monaco", 8))
        self._path_chk.setChecked(True)
        self._path_chk.stateChanged.connect(self._on_path_toggled)
        ll.addWidget(self._path_chk)

        self._grid_chk = QCheckBox("Show reference grid")
        self._grid_chk.setFont(QFont("Monaco", 8))
        self._grid_chk.setChecked(True)
        self._grid_chk.stateChanged.connect(self._on_grid_toggled)
        ll.addWidget(self._grid_chk)

        # Colour legend
        legend = self._build_legend()
        ll.addWidget(self._card(legend, "Distance Colour Key"))

        btn_reconnect = QPushButton("Reconnect")
        btn_reconnect.setFont(QFont("Monaco", 9))
        btn_reconnect.clicked.connect(self._reconnect)
        ll.addWidget(btn_reconnect)

        btn_clear = QPushButton("Clear Map")
        btn_clear.setFont(QFont("Monaco", 9))
        btn_clear.clicked.connect(self._clear_map)
        ll.addWidget(btn_clear)

        btn_reset_cam = QPushButton("Reset Camera")
        btn_reset_cam.setFont(QFont("Monaco", 9))
        btn_reset_cam.clicked.connect(self._on_reset_camera_clicked)
        ll.addWidget(btn_reset_cam)

        btn_auto_pan = QPushButton("Auto Pan")
        btn_auto_pan.setFont(QFont("Monaco", 9))
        btn_auto_pan.clicked.connect(self._toggle_auto_pan)
        ll.addWidget(btn_auto_pan)

        self.auto_pan_status_label = QLabel("Auto-pan off")
        self.auto_pan_status_label.setFont(QFont("Monaco", 8))
        self.auto_pan_status_label.setStyleSheet(f"color: {ACCENT_COLOR}; padding: 4px 0;")
        ll.addWidget(self._card(self.auto_pan_status_label, "Auto Pan Mode"))

        ll.addStretch()

        # ── Centre panel — OpenGL view ────────────────────────────────────────
        centre = QWidget()
        cl = QVBoxLayout()
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        centre.setLayout(cl)

        map_title = QLabel("FIXED-WORLD 3D POINT CLOUD  —  sensor at origin · axes never move")
        map_title.setFont(QFont("Monaco", 10, QFont.Bold))
        map_title.setStyleSheet(f"color: {ACCENT_COLOR}; padding: 2px 0;")
        cl.addWidget(map_title)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor(pg.mkColor(BG_COLOR))
        self.gl_view.setMinimumSize(900, 800)
        cl.addWidget(self.gl_view, 1)

        # ── Right panel — log ─────────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(380)
        rl = QVBoxLayout()
        rl.setContentsMargins(0, 0, 0, 0)
        right.setLayout(rl)

        log_title = QLabel("LIVE SERIAL LOG")
        log_title.setFont(QFont("Monaco", 11, QFont.Bold))
        log_title.setStyleSheet(f"color: {ACCENT_COLOR}; padding: 2px 0;")
        rl.addWidget(log_title)

        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Monaco", 8))
        rl.addWidget(self.log_display)

        root.addWidget(left_scroll,   0)
        root.addWidget(centre, 1)
        root.addWidget(right,  0)

    def _card(self, widget, title_text):
        card = QWidget()
        card.setStyleSheet(f"""
            QWidget {{
                background-color: {CARD_COLOR};
                border: 1px solid {ACCENT_COLOR};
                border-radius: 6px;
                padding: 6px;
                margin: 2px 0px;
            }}
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(2)
        lbl = QLabel(title_text)
        lbl.setFont(QFont("Monaco", 7))
        lbl.setStyleSheet(f"color: {MUTED_COLOR}; border: none;")
        layout.addWidget(lbl)
        layout.addWidget(widget)
        card.setLayout(layout)
        return card

    def _build_legend(self):
        w = QWidget()
        w.setStyleSheet("border: none;")
        hl = QHBoxLayout()
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        w.setLayout(hl)
        entries = [("Near",  "#0000ff"), ("Mid",  "#00ff00"),
                   ("Far",   "#ffff00"), ("Max",  "#ff0000")]
        for label, colour in entries:
            dot = QLabel("●")
            dot.setFont(QFont("Monaco", 14))
            dot.setStyleSheet(f"color: {colour}; border: none;")
            lbl = QLabel(label)
            lbl.setFont(QFont("Monaco", 7))
            lbl.setStyleSheet(f"color: {TEXT_COLOR}; border: none;")
            hl.addWidget(dot)
            hl.addWidget(lbl)
        hl.addStretch()
        return w

    def _apply_styling(self):
        self.setStyleSheet(f"""
            QMainWindow  {{ background-color: {BG_COLOR}; }}
            QWidget      {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
            QLabel        {{ color: {TEXT_COLOR}; }}
            QPlainTextEdit {{
                background-color: {CARD_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
                border-radius: 4px;
                padding: 6px;
                font-family: Monaco;
            }}
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: {TEXT_COLOR};
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-family: Monaco;
                font-weight: bold;
                margin: 3px 0px;
            }}
            QPushButton:hover   {{ background-color: {SUCCESS_COLOR}; }}
            QPushButton:pressed {{ background-color: {WARNING_COLOR}; }}
            QComboBox {{
                background-color: {CARD_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
                border-radius: 4px;
                padding: 4px;
                font-family: Monaco;
            }}
            QCheckBox {{
                color: {TEXT_COLOR};
                font-family: Monaco;
                padding: 4px;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {ACCENT_COLOR};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {SUCCESS_COLOR};
                width: 14px; height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }}
        """)

    # ── OpenGL scene setup ───────────────────────────────────────────────────
    def _build_gl_scene(self):
        """Build all persistent GL items (FR-5: created once, updated after)."""
        self.gl_view.clear()

        half = self._room_half

        # FR-6: Reference grid — XY plane at Z=0 (floor), fixed world coords
        if self._show_grid:
            spacing = 100
            grid_xy = gl.GLGridItem()
            grid_xy.setSize(half * 2, half * 2)
            grid_xy.setSpacing(spacing, spacing)
            grid_xy.setColor(pg.mkColor(60, 90, 160, 120))
            self.gl_view.addItem(grid_xy)

            # XZ wall grid (back wall)
            grid_xz = gl.GLGridItem()
            grid_xz.setSize(half * 2, half)
            grid_xz.setSpacing(spacing, spacing)
            grid_xz.rotate(90, 1, 0, 0)
            grid_xz.translate(0, half, half / 2)
            grid_xz.setColor(pg.mkColor(40, 70, 140, 60))
            self.gl_view.addItem(grid_xz)

        # FR-2: Sensor origin marker — white sphere at (0,0,0)
        origin_data = np.array([[0, 0, 0]], dtype=np.float32)
        origin_color = np.array([[1, 1, 1, 1]], dtype=np.float32)
        self._origin_item = gl.GLScatterPlotItem(
            pos=origin_data, color=origin_color, size=14, pxMode=True
        )
        self.gl_view.addItem(self._origin_item)

        # Axis lines for spatial orientation
        axis_len = half
        for direction, colour in [
            (np.array([[0,0,0],[axis_len,0,0]]), (1.0, 0.3, 0.3, 0.8)),  # X red
            (np.array([[0,0,0],[0,axis_len,0]]), (0.3, 1.0, 0.3, 0.8)),  # Y green
            (np.array([[0,0,0],[0,0,axis_len]]), (0.3, 0.3, 1.0, 0.8)),  # Z blue
        ]:
            ax_item = gl.GLLinePlotItem(
                pos=direction.astype(np.float32),
                color=colour, width=1.5, antialias=True
            )
            self.gl_view.addItem(ax_item)

        # Axis labels via small scatter markers at tips
        label_pts = np.array([
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ], dtype=np.float32)
        label_colors = np.array([
            [1.0, 0.3, 0.3, 1.0],
            [0.3, 1.0, 0.3, 1.0],
            [0.3, 0.3, 1.0, 1.0],
        ], dtype=np.float32)
        self.gl_view.addItem(gl.GLScatterPlotItem(
            pos=label_pts, color=label_colors, size=8, pxMode=True
        ))

        # FR-1: Point cloud scatter — starts empty, updated incrementally
        self._scatter_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.zeros((1, 4), dtype=np.float32),
            size=self._point_size,
            pxMode=True,
        )
        self.gl_view.addItem(self._scatter_item)

        # FR-9: Scan path line — starts empty
        self._path_item = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.3, 0.6, 1.0, 0.5),
            width=1.2,
            antialias=True,
            mode='line_strip',
        )
        self._path_item.setVisible(self._show_path)
        self.gl_view.addItem(self._path_item)

        # Set initial camera (FR-4: user can then move freely)
        self._reset_camera()

    def _reset_camera(self):
        half = self._room_half
        self.gl_view.setCameraPosition(
            distance=half * 3.2,
            elevation=30,
            azimuth=-60,
        )
        self.gl_view.opts['center'] = QVector3D(0, 0, half * 0.3)
        
    def _on_reset_camera_clicked(self):
        now = time.time()
        if now - self._last_reset_press <= self._auto_pan_press_threshold:
            self._toggle_auto_pan()
        else:
            self._reset_camera()
        self._last_reset_press = now

    def _toggle_auto_pan(self):
        self._auto_pan_enabled = not self._auto_pan_enabled
        if self._auto_pan_enabled:
            self._auto_pan_azimuth = float(self.gl_view.opts.get('azimuth', -60))
            self._auto_pan_timer.start()
        else:
            self._auto_pan_timer.stop()
        self._update_auto_pan_status()

    def _advance_auto_pan(self):
        if not self._auto_pan_enabled:
            return
        self._auto_pan_azimuth = (self._auto_pan_azimuth + 0.8) % 360.0
        opts = self.gl_view.opts
        elevation = float(opts.get('elevation', 30))
        distance = float(opts.get('distance', self._room_half * 3.2))
        self.gl_view.setCameraPosition(distance=distance, elevation=elevation, azimuth=self._auto_pan_azimuth)

    def _update_auto_pan_status(self):
        if self.auto_pan_status_label:
            self.auto_pan_status_label.setText("Auto-pan on" if self._auto_pan_enabled else "Auto-pan off")

    # ── Incremental render (FR-5) ─────────────────────────────────────────────
    def _maybe_redraw(self):
        if not self.store.consume_new_flag():
            return
        self._update_scatter()
        self._update_path()
        self._update_object_count_display()

    def _update_scatter(self):
        """FR-5: Replace scatter data only — camera untouched."""
        if self.store.count == 0:
            return
        _, _, dists, xs, ys, zs = self.store.arrays()

        # FR-1/FR-3: Z = distance component from spherical conversion
        pos = np.column_stack([xs, ys, zs]).astype(np.float32)

        # FR-10: colour by distance
        colours = dist_to_rgba(dists, self._room_half * 1.2).astype(np.float32)

        self._scatter_item.setData(
            pos=pos,
            color=colours,
            size=self._point_size,
        )

        # Most-recent point highlighted in white
        if self.last_valid_xyz is not None:
            lx, ly, lz = self.last_valid_xyz
            tip_pos = np.array([[lx, ly, lz]], dtype=np.float32)
            tip_col = np.array([[1, 1, 0, 1]], dtype=np.float32)  # yellow crosshair
            self._origin_item.setData(
                pos=np.vstack([np.array([[0, 0, 0]], dtype=np.float32), tip_pos]),
                color=np.vstack([np.array([[1, 1, 1, 1]], dtype=np.float32), tip_col]),
                size=np.array([14, 10], dtype=np.float32),
            )

        self.total_label.setText(
            f"{self.store.count:,}"
            + (" ⚠ cap" if self.store.count >= MAX_POINT_STORE else "")
        )

    def _update_path(self):
        """FR-9: Rebuild scan path from recent readings buffer."""
        if not self._show_path:
            return
        pts = [
            angles_to_xyz(r['ax'], r['ay'], r['dist'])
            for r in self.readings_buffer if r['dist'] > 0
        ]
        if len(pts) < 2:
            return
        xs, ys, zs = zip(*pts)
        path = np.column_stack([xs, ys, zs]).astype(np.float32)
        self._path_item.setData(pos=path)

    def _load_or_build_object_counter(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            try:
                self._object_counter = joblib.load(str(MODEL_PATH))
                return
            except Exception:
                self._object_counter = None

        self._object_counter = self._build_synthetic_object_counter()

    def _build_synthetic_object_counter(self):
        pan_angles, tilt_angles = self._pan_tilt_grid()
        X = []
        y = []
        for _ in range(OBJECT_COUNTER_TRAIN_SAMPLES):
            count = random.randint(0, 6)
            _, _, distances = self._simulate_scan(count)
            X.append(self._scan_to_features(pan_angles, tilt_angles, distances))
            y.append(count)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)),
        ])
        pipeline.fit(np.vstack(X), np.array(y, dtype=int))
        joblib.dump(pipeline, str(MODEL_PATH))
        return pipeline

    def _pan_tilt_grid(self):
        pan_angles = np.linspace(ANGLE_X_MIN, ANGLE_X_MAX, GRID_PAN_POINTS)
        tilt_angles = np.linspace(ANGLE_Y_MIN, ANGLE_Y_MAX, GRID_TILT_POINTS)
        return pan_angles, tilt_angles

    def _simulate_scan(self, object_count, noise_scale=1.5):
        pan_angles, tilt_angles = self._pan_tilt_grid()
        base_dist = 500.0
        distances = np.full((len(tilt_angles), len(pan_angles)), base_dist, dtype=np.float32)

        for object_index in range(object_count):
            object_pan = random.uniform(ANGLE_X_MIN + 5.0, ANGLE_X_MAX - 5.0)
            object_width = random.uniform(5.0, 18.0)
            object_depth = random.uniform(60.0, 220.0)

            left = object_pan - object_width / 2.0
            right = object_pan + object_width / 2.0

            for i, pan in enumerate(pan_angles):
                if left <= pan <= right:
                    for j, tilt in enumerate(tilt_angles):
                        drop_amount = object_depth * (1.0 - abs((pan - object_pan) / (object_width / 2.0))**2)
                        distances[j, i] = min(distances[j, i], base_dist - drop_amount)

        distances += np.random.normal(scale=noise_scale, size=distances.shape)
        distances = np.clip(distances, 0.0, base_dist)
        return pan_angles, tilt_angles, distances

    def _scan_to_features(self, pan_angles, tilt_angles, distances):
        flat = distances.flatten()
        dx = np.diff(flat)
        large_jumps = np.sum(np.abs(dx) > 12.0)
        small_jumps = np.sum(np.logical_and(np.abs(dx) > 4.0, np.abs(dx) <= 12.0))

        stats = [
            np.mean(flat),
            np.std(flat),
            np.min(flat),
            np.max(flat),
            np.median(flat),
            np.percentile(flat, 10),
            np.percentile(flat, 25),
            np.percentile(flat, 75),
            np.percentile(flat, 90),
            large_jumps,
            small_jumps,
        ]

        hist_bins = np.histogram(flat, bins=[0, 100, 200, 300, 400, 600])[0].astype(float)
        hist_norm = hist_bins / max(np.sum(hist_bins), 1.0)
        return np.concatenate([stats, hist_norm])

    def _buffer_to_features(self):
        if len(self.readings_buffer) < 20:
            return None

        recent = list(self.readings_buffer)[-GRID_PAN_POINTS * GRID_TILT_POINTS :]
        pan_angles, tilt_angles = self._pan_tilt_grid()
        distances = np.full((len(tilt_angles), len(pan_angles)), np.nan, dtype=np.float32)

        for r in recent:
            i = np.searchsorted(pan_angles, r['ax'])
            j = np.searchsorted(tilt_angles, r['ay'])
            if 0 <= i < len(pan_angles) and 0 <= j < len(tilt_angles):
                current = distances[j, i]
                if np.isnan(current) or r['dist'] < current:
                    distances[j, i] = r['dist']

        if np.isnan(distances).all():
            return None
        distances = np.nan_to_num(distances, nan=np.nanmean(distances))
        return self._scan_to_features(pan_angles, tilt_angles, distances)

    def _predict_object_count(self):
        if self._object_counter is None:
            return None
        features = self._buffer_to_features()
        if features is None:
            return None
        try:
            return int(self._object_counter.predict(features.reshape(1, -1))[0])
        except Exception:
            return None

    def _update_object_count_display(self):
        count = self._predict_object_count()
        self.object_count_label.setText(str(count) if count is not None else "—")
        self._update_occupancy_display()
        self._update_stability_display()
        self._update_avg_distance_display()

    def _classify_occupancy(self):
        """Classify room occupancy as empty/sparse/moderate/crowded."""
        if len(self.readings_buffer) < 15:
            return None
        dists = np.array([r['dist'] for r in self.readings_buffer if r['dist'] > 0], dtype=np.float32)
        if len(dists) == 0:
            return "Empty"
        avg_dist = np.mean(dists)
        close_count = np.sum(dists < 200)
        close_ratio = close_count / len(dists)
        if close_ratio > 0.6:
            return "Crowded"
        elif close_ratio > 0.35:
            return "Moderate"
        elif close_ratio > 0.1:
            return "Sparse"
        else:
            return "Empty"

    def _compute_scene_change(self):
        """Compute how much the scene changed from last frame (0-1 scale)."""
        features = self._buffer_to_features()
        if features is None or self._last_scan_features is None:
            return None
        try:
            diff = np.linalg.norm(features - self._last_scan_features)
            normalized = min(diff / 50.0, 1.0)
            return normalized
        except Exception:
            return None

    def _compute_avg_distance(self):
        """Return average distance in the current buffer."""
        if len(self.readings_buffer) < 10:
            return None
        dists = np.array([r['dist'] for r in self.readings_buffer if r['dist'] > 0], dtype=np.float32)
        if len(dists) == 0:
            return None
        return np.mean(dists)

    def _update_occupancy_display(self):
        occupancy = self._classify_occupancy()
        self.occupancy_label.setText(occupancy if occupancy else "—")

    def _update_stability_display(self):
        change = self._compute_scene_change()
        if change is None:
            self.stability_label.setText("—")
        else:
            stability_pct = int((1.0 - change) * 100)
            self.stability_label.setText(f"{stability_pct}%")
            self._last_scan_features = self._buffer_to_features()

    def _update_avg_distance_display(self):
        avg = self._compute_avg_distance()
        if avg is None:
            self.avg_dist_label.setText("—")
        else:
            self.avg_dist_label.setText(f"{avg:.0f} cm")

    # ── Control callbacks ─────────────────────────────────────────────────────
    def _on_room_changed(self, name: str):
        self._room_half = ROOM_PRESETS.get(name, 500)
        self._build_gl_scene()
        # Replot existing data
        self.store._new_since_render = self.store.count  # force full refresh
        self.store.consume_new_flag()
        self._update_scatter()
        self._update_path()

    def _on_point_size_changed(self, v: int):
        self._point_size = float(v)
        self._ps_label.setText(f"Point size: {v}")
        if self._scatter_item:
            self._scatter_item.setData(size=self._point_size)

    def _on_path_toggled(self, state):
        self._show_path = bool(state)
        if self._path_item:
            self._path_item.setVisible(self._show_path)

    def _on_grid_toggled(self, state):
        self._show_grid = bool(state)
        self._build_gl_scene()
        self.store._new_since_render = self.store.count
        self.store.consume_new_flag()
        self._update_scatter()
        self._update_path()

    # ── Serial callbacks ──────────────────────────────────────────────────────
    def _on_line_received(self, line: str):
        self.line_history.append(line)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_display.appendPlainText(f"[{ts}] {line}")
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_display.setTextCursor(cursor)
        self._parse_line(line)

    def _parse_line(self, line: str):
        self.raw_label.setText(line[:100] + ("..." if len(line) > 100 else ""))
        m = re.search(
            r"(-?\d+(?:\.\d*)?)\s*,\s*(-?\d+(?:\.\d*)?)\s*,\s*(-?\d+(?:\.\d*)?)", line
        )
        if not m:
            return
        try:
            ax_deg = float(m.group(1))
            ay_deg = float(m.group(2))
            dist   = float(m.group(3))
        except ValueError:
            return

        self.pan_label.setText(f"{ax_deg:.6f}°")
        self.tilt_label.setText(f"{ay_deg:.6f}°")

        # ── Previous-distance lookup & delta ──────────────────────────────
        # Quantize to the same cell resolution the Arduino uses (1°) so the
        # "previous" value corresponds to the same physical beam direction.
        cell_key = _angle_key(ax_deg, ay_deg)
        prev_dist = self._prev_dist_map.get(cell_key)  # None on first visit

        if dist > 0.0:
            # Current distance display
            self.dist_label.setText(f"{dist:.2f} cm")

            if prev_dist is not None:
                delta = dist - prev_dist                    # + means moved away
                sign  = "+" if delta >= 0 else ""
                abs_d = abs(delta)

                self.prev_dist_label.setText(f"{prev_dist:.2f} cm")

                if abs_d < 1.0:
                    colour = DELTA_NEU_COLOR
                    arrow  = "≈"
                elif delta < 0:
                    colour = DELTA_POS_COLOR   # object closer → green
                    arrow  = "▲"
                else:
                    colour = DELTA_NEG_COLOR   # object farther → orange
                    arrow  = "▼"

                self.delta_dist_label.setText(f"{arrow} {sign}{delta:.2f} cm")
                self.delta_dist_label.setStyleSheet(
                    f"color: {colour}; border: none; font-family: Monaco;"
                )
            else:
                # First visit to this cell — no previous reading yet
                self.prev_dist_label.setText("(first reading)")
                self.delta_dist_label.setText("—")
                self.delta_dist_label.setStyleSheet(
                    f"color: {DELTA_NEU_COLOR}; border: none; font-family: Monaco;"
                )

            # Update the map so the NEXT visit to this cell can compare
            self._prev_dist_map[cell_key] = dist

        else:
            self.dist_label.setText("No echo")
            self.prev_dist_label.setText(f"{prev_dist:.2f} cm" if prev_dist is not None else "—")
            self.delta_dist_label.setText("—")
            self.delta_dist_label.setStyleSheet(
                f"color: {DELTA_NEU_COLOR}; border: none; font-family: Monaco;"
            )

        self.readings_buffer.append({"ax": ax_deg, "ay": ay_deg,
                                     "dist": dist, "t": time.time()})

        in_range = (
            dist > 0.0
            and (ANGLE_X_MIN - 1.0) <= ax_deg <= (ANGLE_X_MAX + 1.0)
            and (ANGLE_Y_MIN - 1.0) <= ay_deg <= (ANGLE_Y_MAX + 1.0)
        )
        if in_range:
            self.store.add(ax_deg, ay_deg, dist)
            self.last_valid_xyz = angles_to_xyz(ax_deg, ay_deg, dist)

    def _on_connected(self):
        self.status_label.setText("● CONNECTED")
        self.status_label.setStyleSheet(f"color: {SUCCESS_COLOR}; padding: 8px;")
        self.log_display.appendPlainText("✓ Serial connection established")

    def _on_disconnected(self):
        self.status_label.setText("● DISCONNECTED")
        self.status_label.setStyleSheet(f"color: {WARNING_COLOR}; padding: 8px;")

    def _on_connection_failed(self, error: str):
        self.status_label.setText("● ERROR")
        self.status_label.setStyleSheet(f"color: {WARNING_COLOR}; padding: 8px;")
        self.log_display.appendPlainText(f"✗ Connection failed: {error}")

    def _start_serial_connection(self):
        self.serial_thread = SerialReaderThread(PORT, BAUD)
        self.serial_thread.line_received.connect(self._on_line_received)
        self.serial_thread.connected.connect(self._on_connected)
        self.serial_thread.disconnected.connect(self._on_disconnected)
        self.serial_thread.connection_failed.connect(self._on_connection_failed)
        self.serial_thread.start()

    def _reconnect(self):
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait()
        self._start_serial_connection()

    def _clear_map(self):
        self.store.clear()
        self.readings_buffer.clear()
        self.last_valid_xyz = None
        self._prev_dist_map.clear()          # reset angle → distance history
        self.total_label.setText("0")
        self.prev_dist_label.setText("—")
        self.delta_dist_label.setText("—")
        self.delta_dist_label.setStyleSheet(
            f"color: {DELTA_NEU_COLOR}; border: none; font-family: Monaco;"
        )
        self.log_display.appendPlainText("✓ Map cleared")
        # Reset scatter to empty
        if self._scatter_item:
            self._scatter_item.setData(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=np.zeros((1, 4), dtype=np.float32),
            )
        if self._path_item:
            self._path_item.setData(pos=np.zeros((1, 3), dtype=np.float32))

    def closeEvent(self, event):
        self._redraw_timer.stop()
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    pg.setConfigOptions(antialias=True)
    app    = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()