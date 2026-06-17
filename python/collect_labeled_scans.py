"""Collect labeled scan frames from the Teensy serial stream.

This script reads the same comma-delimited lines as visualizer.py and writes
complete scan frames to data/labeled_scans.csv.

The CSV format is:
    frame_id,pan,tilt,dist,label

Example:
    1,40.0,0.0,120.4,2
    1,42.0,0.0,118.7,2

A new frame is detected when pan resets from a high angle back toward the start
or when the serial stream pauses.

Usage:
    python python/collect_labeled_scans.py
"""

import csv
import os
import re
import time

import serial

PORT = "/dev/cu.usbmodem111427801"
BAUD = 9600
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "labeled_scans.csv")
FRAME_RESET_THRESHOLD = 15.0
TIMEOUT_SECONDS = 0.8
MIN_FRAME_POINTS = 10

LINE_RE = re.compile(r"(-?\d+(?:\.\d*)?)\s*,\s*(-?\d+(?:\.\d*)?)\s*,\s*(-?\d+(?:\.\d*)?)")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def parse_line(line: str):
    m = LINE_RE.search(line)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    except ValueError:
        return None


def append_frame(frame_id, rows, label):
    ensure_data_dir()
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["frame_id", "pan", "tilt", "dist", "label"])
        for pan, tilt, dist in rows:
            writer.writerow([frame_id, f"{pan:.6f}", f"{tilt:.6f}", f"{dist:.6f}", label])


def prompt_label(frame_id, frame_size):
    while True:
        user = input(f"Frame {frame_id} complete ({frame_size} points). Enter object count label: ")
        if user.strip() == "":
            print("Label required. Please enter a number.")
            continue
        if user.isdigit() and int(user) >= 0:
            return int(user)
        print("Invalid label. Enter a non-negative integer.")


def main():
    ensure_data_dir()
    print("Starting labeled scan collection")
    print(f"Port: {PORT}, Baud: {BAUD}")
    print(f"Saving labeled frames to: {CSV_PATH}")
    print("Press Ctrl+C to stop.")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except serial.SerialException as exc:
        print(f"Failed to open serial port: {exc}")
        return

    frame_id = 1
    current_frame = []
    last_pan = None
    last_time = time.time()

    try:
        while True:
            raw = ser.readline()
            if not raw:
                if current_frame and time.time() - last_time > TIMEOUT_SECONDS:
                    if len(current_frame) >= MIN_FRAME_POINTS:
                        label = prompt_label(frame_id, len(current_frame))
                        append_frame(frame_id, current_frame, label)
                        frame_id += 1
                    current_frame = []
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            parsed = parse_line(line)
            if parsed is None:
                continue

            pan, tilt, dist = parsed
            now = time.time()
            reset_detected = False
            if last_pan is not None and pan < last_pan - FRAME_RESET_THRESHOLD:
                reset_detected = True
            elif now - last_time > TIMEOUT_SECONDS:
                reset_detected = True

            if reset_detected and current_frame:
                if len(current_frame) >= MIN_FRAME_POINTS:
                    label = prompt_label(frame_id, len(current_frame))
                    append_frame(frame_id, current_frame, label)
                    frame_id += 1
                current_frame = []

            current_frame.append((pan, tilt, dist))
            last_pan = pan
            last_time = now
    except KeyboardInterrupt:
        print("\nStopping collection.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
