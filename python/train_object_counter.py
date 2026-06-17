"""Train a small object-count model for room scan data.

This script can train from either:
- a labeled CSV dataset at data/labeled_scans.csv
- a synthetic training set generated from simulated object scans

Expected labeled CSV format:
frame_id,pan,tilt,dist,label
1,40.0,0.0,120.4,2
1,42.0,0.0,118.7,2
... and so on.

If no labeled CSV is present, the script uses synthetic scan generation to demonstrate a working model.

Usage:
    python python/train_object_counter.py
"""

import argparse
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

PAN_MIN = 40.0
PAN_MAX = 180.0
TILT_MIN = 0.0
TILT_MAX = 30.0
NUM_PAN_POINTS = 35
NUM_TILT_POINTS = 6


def make_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def pan_tilt_grid():
    pan_angles = np.linspace(PAN_MIN, PAN_MAX, NUM_PAN_POINTS)
    tilt_angles = np.linspace(TILT_MIN, TILT_MAX, NUM_TILT_POINTS)
    return pan_angles, tilt_angles


def simulate_scan(object_count, noise_scale=1.5):
    """Simulate a room scan with a given number of foreground objects."""
    pan_angles, tilt_angles = pan_tilt_grid()
    base_dist = 500.0
    distances = np.full((len(tilt_angles), len(pan_angles)), base_dist, dtype=np.float32)

    for object_index in range(object_count):
        object_pan = random.uniform(PAN_MIN + 5.0, PAN_MAX - 5.0)
        object_width = random.uniform(5.0, 18.0)
        object_depth = random.uniform(60.0, 220.0)
        object_height = random.uniform(40.0, 120.0)

        object_dist = base_dist - object_depth
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


def scan_to_features(pan_angles, tilt_angles, distances):
    """Extract features from a single scan grid."""
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


def load_labeled_dataset(data_path):
    if not os.path.isfile(data_path):
        return None

    df = pd.read_csv(data_path)
    if 'label' not in df.columns:
        raise ValueError('Labeled dataset must include a "label" column.')

    features = []
    labels = []

    for frame_id, frame_group in df.groupby('frame_id'):
        pan = frame_group['pan'].values
        tilt = frame_group['tilt'].values
        dist = frame_group['dist'].values
        if len(pan) == 0:
            continue
        scan = pd.DataFrame({'pan': pan, 'tilt': tilt, 'dist': dist})
        pan_grid = np.sort(scan['pan'].unique())
        tilt_grid = np.sort(scan['tilt'].unique())
        distances = np.full((len(tilt_grid), len(pan_grid)), np.nan, dtype=np.float32)
        for _, row in scan.iterrows():
            i = np.searchsorted(pan_grid, row['pan'])
            j = np.searchsorted(tilt_grid, row['tilt'])
            if 0 <= i < len(pan_grid) and 0 <= j < len(tilt_grid):
                distances[j, i] = row['dist']

        if np.isnan(distances).any():
            distances = np.nan_to_num(distances, nan=np.nanmean(distances))

        features.append(scan_to_features(pan_grid, tilt_grid, distances))
        labels.append(int(frame_group['label'].iloc[0]))

    return np.vstack(features), np.array(labels, dtype=int)


def generate_synthetic_dataset(samples=600, max_objects=6):
    data = []
    labels = []
    for _ in range(samples):
        count = random.randint(0, max_objects)
        pan, tilt, distances = simulate_scan(count)
        data.append(scan_to_features(pan, tilt, distances))
        labels.append(count)
    return np.vstack(data), np.array(labels, dtype=int)


def train_model(X, y, model_path):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print('Model evaluation:')
    print('------------------')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'MAE count error: {mean_absolute_error(y_test, y_pred):.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipeline, model_path)
    print(f'✅ Saved trained model to {model_path}')
    return pipeline


def main():
    parser = argparse.ArgumentParser(description='Train an object-count model from room sweep scans.')
    parser.add_argument('--dataset', default=os.path.join(DATA_DIR, 'labeled_scans.csv'), help='Labeled CSV dataset path')
    parser.add_argument('--save-model', default=os.path.join(MODEL_DIR, 'object_counter.pkl'), help='Model output path')
    parser.add_argument('--synthetic', action='store_true', help='Force synthetic data generation instead of loading CSV')
    parser.add_argument('--samples', type=int, default=800, help='Synthetic samples to generate when using synthetic mode')
    args = parser.parse_args()

    make_dirs()

    if not args.synthetic:
        dataset = load_labeled_dataset(args.dataset)
        if dataset is not None:
            X, y = dataset
            print(f'Loaded labeled dataset: {X.shape[0]} frames, {len(np.unique(y))} label classes')
        else:
            print('No labeled dataset found, switching to synthetic data.')
            args.synthetic = True

    if args.synthetic:
        X, y = generate_synthetic_dataset(samples=args.samples)
        print(f'Generated synthetic dataset: {X.shape[0]} samples')

    train_model(X, y, args.save_model)

    # Example inference on a synthetic scan
    pan, tilt, distances = simulate_scan(object_count=random.randint(0, 5))
    features = scan_to_features(pan, tilt, distances).reshape(1, -1)
    pipeline = joblib.load(args.save_model)
    prediction = pipeline.predict(features)[0]
    print(f'Example inference result: predicted object count = {prediction}')


if __name__ == '__main__':
    main()
