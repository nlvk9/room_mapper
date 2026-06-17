#pragma once
#include <stdint.h>

// ---------------------------------------------------------------------------
// KalmanGrid — per-cell 1D Kalman filter for the room topology scanner
//
// Each (x_angle, y_angle) grid cell maintains two floats:
//   estimate  — current best guess of true distance (mm)
//   errorCov  — current estimation error covariance
//
// Tuning parameters (adjust to taste):
//   PROCESS_NOISE   (Q) — how much we expect the surface to "drift" per scan.
//                         Lower = smoother but slower to track real changes.
//   MEASURE_NOISE   (R) — how noisy we believe the HC-SR04 is (mm²).
//                         Higher = smoother but less responsive.
//   MOTION_THRESH       — delta (mm) between smoothed frames that fires a
//                         MOTION event. Tune based on your room size.
// ---------------------------------------------------------------------------

static constexpr float PROCESS_NOISE = 1.0f;  // Q  (mm²)
static constexpr float MEASURE_NOISE = 50.0f; // R  (mm²) — HC-SR04 is ~3mm std dev
static constexpr float MOTION_THRESH = 30.0f; // mm — minimum delta to fire MOTION
static constexpr float MAX_DIST_MM = 400.0f;

// Grid dimensions — must match C++ constants in src/uno/main.cpp
static constexpr int X_MIN = 25, X_MAX = 105; // servo degrees
static constexpr int Y_MIN = 130, Y_MAX = 180;

static constexpr int GRID_W = X_MAX - X_MIN + 1; // 81 cells
static constexpr int GRID_H = Y_MAX - Y_MIN + 1; // 51 cells
static constexpr int GRID_N = GRID_W * GRID_H;   // 4131 cells
                                                 // × 2 floats = ~32 KB — fits in 64 KB RAM

struct KalmanCell
{
    float estimate; // mm
    float errorCov; // initialised high so first reading is trusted fully
};

class KalmanGrid
{
public:
    KalmanGrid() { reset(); }

    void reset()
    {
        for (int i = 0; i < GRID_N; i++)
        {
            _cells[i] = {MAX_DIST_MM, 1000.0f}; // high initial uncertainty
        }
        for (int i = 0; i < GRID_N; i++)
        {
            _prev[i] = MAX_DIST_MM;
        }
    }

    // Update one cell with a new raw measurement.
    // Returns the smoothed estimate after the update.
    float update(int x_deg, int y_deg, float measured_mm)
    {
        int idx = index(x_deg, y_deg);
        if (idx < 0)
            return measured_mm;

        KalmanCell &c = _cells[idx];

        // Predict step — error covariance grows by process noise each cycle
        float p_pred = c.errorCov + PROCESS_NOISE;

        // Update step
        float K = p_pred / (p_pred + MEASURE_NOISE); // Kalman gain
        c.estimate = c.estimate + K * (measured_mm - c.estimate);
        c.errorCov = (1.0f - K) * p_pred;

        return c.estimate;
    }

    // Check whether smoothed estimate has changed enough vs. previous frame.
    // Returns true if a MOTION event should be emitted; updates prev frame.
    bool checkMotion(int x_deg, int y_deg)
    {
        int idx = index(x_deg, y_deg);
        if (idx < 0)
            return false;

        float current = _cells[idx].estimate;
        float delta = current - _prev[idx];
        if (delta < 0)
            delta = -delta;

        bool moved = (delta > MOTION_THRESH);
        _prev[idx] = current; // always update previous frame
        return moved;
    }

    float getEstimate(int x_deg, int y_deg) const
    {
        int idx = index(x_deg, y_deg);
        return (idx >= 0) ? _cells[idx].estimate : MAX_DIST_MM;
    }

private:
    KalmanCell _cells[GRID_N];
    float _prev[GRID_N];

    static int index(int x_deg, int y_deg)
    {
        int xi = x_deg - X_MIN;
        int yi = y_deg - Y_MIN;
        if (xi < 0 || xi >= GRID_W || yi < 0 || yi >= GRID_H)
            return -1;
        return yi * GRID_W + xi;
    }
};
