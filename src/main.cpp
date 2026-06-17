/* libraries */
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <algorithm> // std::min
#include <cstring>   // std::strcmp
#include <cmath>     // fabsf, round

/* constants */
// PWM for servos
constexpr uint16_t SERVO_MIN_PULSE = 150;
constexpr uint16_t SERVO_MAX_PULSE = 600;

// angle limits
constexpr double LIMIT_X_MIN_ANGLE = 40.0;
constexpr double LIMIT_X_MAX_ANGLE = 180.0;
constexpr double LIMIT_Y_MIN_ANGLE = 0.0;
constexpr double LIMIT_Y_MAX_ANGLE = 30.0;

// PCA9685 channels
constexpr uint8_t SERVO_X_CHANNEL = 4;
constexpr uint8_t SERVO_Y_CHANNEL = 0;

// pins
constexpr uint8_t LED_PIN = 7;
constexpr uint8_t ALERT_LED_PIN = 11; // object-detected indicator
constexpr uint8_t BUTTON_PIN = 12;
constexpr uint8_t TRIG_PIN = 3;
constexpr uint8_t ECHO_PIN = 1;
constexpr uint8_t POT_X_PIN = 14;
constexpr uint8_t POT_Y_PIN = 15;
constexpr uint8_t POT_SPEED_PIN = 20;

// button logic levels
constexpr int NOT_PRESSED = HIGH;
constexpr int PRESSED = LOW;

// EMA filter & distance timing
constexpr float FILTER_WEIGHT = 0.30f;
constexpr unsigned long DISTANCE_INTERVAL = 60UL; // ms

// Teensy 13-bit ADC full scale (0–8191)
constexpr float ADC_FULL_SCALE = 8191.0f;

// speed step range
constexpr float MAX_STEP_MIN = 1.0f;
constexpr float MAX_STEP_MAX = 5.0f;

// HC-SR04
constexpr unsigned long ECHO_TIMEOUT_US = 30000UL;
constexpr float NO_ECHO_SENTINEL = -1.0f;

// fixed environment values for speed-of-sound
constexpr float DEFAULT_TEMPERATURE_C = 20.0f;
constexpr float DEFAULT_HUMIDITY_RH = 50.0f;

// ─────────────────────────────────────────────────────────────────────────────
//  CHANGE-DETECTION GRID
//
//  Each (pan, tilt) angle pair is rounded to the nearest 1.0° and used as
//  the grid key.  The grid stores one float distance per cell.  When a new
//  valid reading arrives at a cell that already has a baseline:
//    • if |new − stored| ≥ CHANGE_THRESHOLD_CM  → alert LED turns on
//    • otherwise the alert LED turns off immediately
//    • the stored value is always updated to the latest reading
//
//  Angle rounding is ONLY used for the grid lookup.  The servo pulse writes
//  always use full-precision floats so motion quality is unaffected.
// ─────────────────────────────────────────────────────────────────────────────

// Distance change required to trigger the alert (cm)
constexpr float CHANGE_THRESHOLD_CM = 25.0f;

// Angular resolution of each grid cell (degrees).
constexpr float CELL_DEG = 1.0f;

// Grid dimensions derived from angle ranges and cell size
constexpr int GRID_X = static_cast<int>((LIMIT_X_MAX_ANGLE - LIMIT_X_MIN_ANGLE) / CELL_DEG) + 1;
constexpr int GRID_Y = static_cast<int>((LIMIT_Y_MAX_ANGLE - LIMIT_Y_MIN_ANGLE) / CELL_DEG) + 1;

// Sentinel meaning "no reading stored yet for this cell"
constexpr float CELL_EMPTY = -1.0f;

// ─────────────────────────────────────────────────────────────────────────────
//  MUTABLE RUNTIME STATE
// ─────────────────────────────────────────────────────────────────────────────
struct State
{
    // Potentiometer raw & filtered values
    int raw_pot_x{0};
    int raw_pot_y{0};
    float filtered_pot_x{4096.0f};
    float filtered_pot_y{4096.0f};
    float filtered_pot_speed{4096.0f};

    // Servo pulse positions (float for sub-step smoothing)
    float current_pulse_x{0.0f};
    float current_pulse_y{0.0f};

    // Target angles — kept for telemetry only, not used for grid lookup
    double current_angle_x{40.0};
    double current_angle_y{15.0};

    // Derived speed step
    float max_step{MAX_STEP_MIN};

    // Button
    int button_state{NOT_PRESSED};
    int prev_button_state{NOT_PRESSED};

    // Auto-pan state
    bool auto_pan_active{false};
    double auto_pan_angle{LIMIT_X_MIN_ANGLE};
    int auto_pan_direction{1};
    double auto_pan_step_deg{1.5};

    // Distance (cm); NO_ECHO_SENTINEL when no valid echo
    float global_distance{NO_ECHO_SENTINEL};

    // Timing
    unsigned long last_distance_time{0UL};

    // ── Change-detection grid ───────────────────────────────────────────────
    float cell_dist[GRID_Y][GRID_X];

    State()
    {
        for (int j = 0; j < GRID_Y; ++j)
            for (int i = 0; i < GRID_X; ++i)
                cell_dist[j][i] = CELL_EMPTY;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  GLOBALS
// ─────────────────────────────────────────────────────────────────────────────
static Adafruit_PWMServoDriver pwm;
static State g;

// ─────────────────────────────────────────────────────────────────────────────
//  FORWARD DECLARATIONS
// ─────────────────────────────────────────────────────────────────────────────
static double angleToPulse(double angle);
static double pulseToAngle(float pulse);
static void servo_function(State &s);
static void servo_auto_pan(State &s);
static float read_distance();
static void update_inputs_and_filters(State &s);
static void print_system_status(const State &s, const char *mode);
static bool i2c_check();
static void update_change_detection(State &s, float dist_cm);

// ─────────────────────────────────────────────────────────────────────────────
//  setup()
// ─────────────────────────────────────────────────────────────────────────────
void setup()
{
    Serial.begin(9600);

    pinMode(LED_PIN, OUTPUT);
    pinMode(ALERT_LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(POT_X_PIN, INPUT);
    pinMode(POT_Y_PIN, INPUT);
    pinMode(POT_SPEED_PIN, INPUT);

    analogReadResolution(13); // 0–8191

    Wire.begin();
    pwm.begin();
    pwm.setPWMFreq(50);

    constexpr float START_ANGLE_X = 40.0f;
    constexpr float START_ANGLE_Y = 15.0f;

    g.current_pulse_x = static_cast<float>(angleToPulse(START_ANGLE_X));
    g.current_pulse_y = static_cast<float>(angleToPulse(START_ANGLE_Y));

    pwm.setPWM(SERVO_X_CHANNEL, 0, static_cast<uint16_t>(g.current_pulse_x));
    pwm.setPWM(SERVO_Y_CHANNEL, 0, static_cast<uint16_t>(g.current_pulse_y));

    delay(1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  loop()
// ─────────────────────────────────────────────────────────────────────────────
void loop()
{
    const bool device_found = i2c_check();

    Serial.print("I2C ");
    Serial.print(device_found ? "Ok | " : "Fail | ");

    g.button_state = digitalRead(BUTTON_PIN);

    // Toggle auto-pan on rising edge
    if (g.button_state == PRESSED && g.prev_button_state == NOT_PRESSED)
    {
        g.auto_pan_active = !g.auto_pan_active;
        if (g.auto_pan_active)
        {
            g.auto_pan_angle = LIMIT_X_MIN_ANGLE;
            g.auto_pan_direction = 1;
        }
        delay(50); // debounce
    }

    update_inputs_and_filters(g);

    // Non-blocking distance sample
    if (millis() - g.last_distance_time >= DISTANCE_INTERVAL)
    {
        g.global_distance = read_distance();
        g.last_distance_time = millis();

        if (g.global_distance > 0.0f)
            update_change_detection(g, g.global_distance);
    }

    if (g.auto_pan_active)
    {
        servo_auto_pan(g);
        print_system_status(g, "AUTO");
        digitalWrite(LED_PIN, HIGH);
    }
    else
    {
        servo_function(g);
        print_system_status(g, "MANUAL");
        digitalWrite(LED_PIN, LOW);
    }

    g.prev_button_state = g.button_state;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CHANGE DETECTION
//
//  1. Back-derive the actual current angle from the servo pulse width.
//  2. Round both axes to the nearest CELL_DEG → grid cell key.
//  3. First visit to a cell: store baseline, LED off, no alert.
//  4. Subsequent visits: LED on if |new − stored| ≥ CHANGE_THRESHOLD_CM,
//     off otherwise.  Always update stored value.
// ─────────────────────────────────────────────────────────────────────────────
static void update_change_detection(State &s, float dist_cm)
{
    const double raw_pan_deg = pulseToAngle(s.current_pulse_x);
    const double raw_tilt_deg = pulseToAngle(s.current_pulse_y);

    const double pan_deg = round(raw_pan_deg);
    const double tilt_deg = round(raw_tilt_deg);

    int ix = static_cast<int>((pan_deg - LIMIT_X_MIN_ANGLE) / CELL_DEG);
    int iy = static_cast<int>((tilt_deg - LIMIT_Y_MIN_ANGLE) / CELL_DEG);

    ix = constrain(ix, 0, GRID_X - 1);
    iy = constrain(iy, 0, GRID_Y - 1);

    float &stored = s.cell_dist[iy][ix];

    if (stored == CELL_EMPTY)
    {
        stored = dist_cm;
        digitalWrite(ALERT_LED_PIN, LOW);
        return;
    }

    const float delta = fabsf(stored - dist_cm);
    digitalWrite(ALERT_LED_PIN, delta >= CHANGE_THRESHOLD_CM ? HIGH : LOW);

    stored = dist_cm;
}

// ─────────────────────────────────────────────────────────────────────────────
//  HELPER: angle → PCA9685 pulse-width count
// ─────────────────────────────────────────────────────────────────────────────
static double angleToPulse(double angle)
{
    return SERVO_MIN_PULSE +
           (angle / 180.0) * static_cast<double>(SERVO_MAX_PULSE - SERVO_MIN_PULSE);
}

// ─────────────────────────────────────────────────────────────────────────────
//  HELPER: PCA9685 pulse-width count → angle (exact inverse of angleToPulse)
// ─────────────────────────────────────────────────────────────────────────────
static double pulseToAngle(float pulse)
{
    return (static_cast<double>(pulse) - SERVO_MIN_PULSE) /
           static_cast<double>(SERVO_MAX_PULSE - SERVO_MIN_PULSE) * 180.0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  SERVO MOTION — potentiometer-controlled
// ─────────────────────────────────────────────────────────────────────────────
static void servo_function(State &s)
{
    const double target_angle_x =
        LIMIT_X_MIN_ANGLE +
        (s.filtered_pot_x / ADC_FULL_SCALE) * (LIMIT_X_MAX_ANGLE - LIMIT_X_MIN_ANGLE);

    const double target_angle_y =
        LIMIT_Y_MIN_ANGLE +
        (s.filtered_pot_y / ADC_FULL_SCALE) * (LIMIT_Y_MAX_ANGLE - LIMIT_Y_MIN_ANGLE);

    const double target_pulse_x = angleToPulse(target_angle_x);
    const double target_pulse_y = angleToPulse(target_angle_y);

    if (s.current_pulse_x < target_pulse_x)
        s.current_pulse_x += std::min<float>(s.max_step, target_pulse_x - s.current_pulse_x);
    else if (s.current_pulse_x > target_pulse_x)
        s.current_pulse_x -= std::min<float>(s.max_step, s.current_pulse_x - target_pulse_x);

    if (s.current_pulse_y < target_pulse_y)
        s.current_pulse_y += std::min<float>(s.max_step, target_pulse_y - s.current_pulse_y);
    else if (s.current_pulse_y > target_pulse_y)
        s.current_pulse_y -= std::min<float>(s.max_step, s.current_pulse_y - target_pulse_y);

    s.current_angle_x = target_angle_x;
    s.current_angle_y = target_angle_y;

    pwm.setPWM(SERVO_X_CHANNEL, 0, static_cast<uint16_t>(s.current_pulse_x));
    pwm.setPWM(SERVO_Y_CHANNEL, 0, static_cast<uint16_t>(s.current_pulse_y));

    delay(15);
}

// ─────────────────────────────────────────────────────────────────────────────
//  SERVO MOTION — auto-pan sweep
// ─────────────────────────────────────────────────────────────────────────────
static void servo_auto_pan(State &s)
{
    const double step = s.auto_pan_step_deg * (s.max_step / MAX_STEP_MAX);

    s.auto_pan_angle += s.auto_pan_direction * step;
    if (s.auto_pan_angle >= LIMIT_X_MAX_ANGLE)
    {
        s.auto_pan_angle = LIMIT_X_MAX_ANGLE;
        s.auto_pan_direction = -1;
    }
    else if (s.auto_pan_angle <= LIMIT_X_MIN_ANGLE)
    {
        s.auto_pan_angle = LIMIT_X_MIN_ANGLE;
        s.auto_pan_direction = 1;
    }

    const double target_angle_y =
        LIMIT_Y_MIN_ANGLE +
        (s.filtered_pot_y / ADC_FULL_SCALE) * (LIMIT_Y_MAX_ANGLE - LIMIT_Y_MIN_ANGLE);

    const double target_pulse_x = angleToPulse(s.auto_pan_angle);
    const double target_pulse_y = angleToPulse(target_angle_y);

    if (s.current_pulse_x < target_pulse_x)
        s.current_pulse_x += std::min<float>(s.max_step, target_pulse_x - s.current_pulse_x);
    else if (s.current_pulse_x > target_pulse_x)
        s.current_pulse_x -= std::min<float>(s.max_step, s.current_pulse_x - target_pulse_x);

    if (s.current_pulse_y < target_pulse_y)
        s.current_pulse_y += std::min<float>(s.max_step, target_pulse_y - s.current_pulse_y);
    else if (s.current_pulse_y > target_pulse_y)
        s.current_pulse_y -= std::min<float>(s.max_step, s.current_pulse_y - target_pulse_y);

    s.current_angle_x = s.auto_pan_angle;
    s.current_angle_y = target_angle_y;

    pwm.setPWM(SERVO_X_CHANNEL, 0, static_cast<uint16_t>(s.current_pulse_x));
    pwm.setPWM(SERVO_Y_CHANNEL, 0, static_cast<uint16_t>(s.current_pulse_y));

    delay(15);
}

// ─────────────────────────────────────────────────────────────────────────────
//  HC-SR04 DISTANCE READING
// ─────────────────────────────────────────────────────────────────────────────
static float read_distance()
{
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    const unsigned long duration = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
    if (duration == 0UL)
        return NO_ECHO_SENTINEL;

    const float v_dry_ms = 331.3f * sqrtf(1.0f + DEFAULT_TEMPERATURE_C / 273.15f);
    const float p_sat_hpa = 6.1078f *
                            powf(10.0f, (7.5f * DEFAULT_TEMPERATURE_C) /
                                            (237.3f + DEFAULT_TEMPERATURE_C));
    const float x_w = (DEFAULT_HUMIDITY_RH / 100.0f) * (p_sat_hpa / 1013.25f);
    const float v_cms_us = v_dry_ms * (1.0f + 0.16f * x_w) / 10000.0f;

    return (static_cast<float>(duration) * v_cms_us) / 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  POTENTIOMETER READ + EMA FILTER
// ─────────────────────────────────────────────────────────────────────────────
static void update_inputs_and_filters(State &s)
{
    s.raw_pot_x = analogRead(POT_X_PIN);
    s.raw_pot_y = analogRead(POT_Y_PIN);
    const int raw_pot_speed = analogRead(POT_SPEED_PIN);

    constexpr float INV_WEIGHT = 1.0f - FILTER_WEIGHT;

    s.filtered_pot_x = (s.filtered_pot_x * INV_WEIGHT) +
                       (static_cast<float>(s.raw_pot_x) * FILTER_WEIGHT);
    s.filtered_pot_y = (s.filtered_pot_y * INV_WEIGHT) +
                       (static_cast<float>(s.raw_pot_y) * FILTER_WEIGHT);
    s.filtered_pot_speed = (s.filtered_pot_speed * INV_WEIGHT) +
                           (static_cast<float>(raw_pot_speed) * FILTER_WEIGHT);

    s.max_step = MAX_STEP_MIN +
                 (s.filtered_pot_speed / ADC_FULL_SCALE) * (MAX_STEP_MAX - MAX_STEP_MIN);
}

// ─────────────────────────────────────────────────────────────────────────────
//  TELEMETRY OUTPUT
// ─────────────────────────────────────────────────────────────────────────────
static void print_system_status(const State &s, const char *mode)
{
    const double pan_deg = round(pulseToAngle(s.current_pulse_x) * 10.0) / 10.0;
    const double tilt_deg = round(pulseToAngle(s.current_pulse_y) * 10.0) / 10.0;

    Serial.print(mode);
    Serial.print(" | ");
    Serial.print(static_cast<int>(s.filtered_pot_x));
    Serial.print(" | ");
    Serial.print(static_cast<int>(s.filtered_pot_y));
    Serial.print(" | ");
    Serial.print(pan_deg, 1);
    Serial.print(",");
    Serial.print(tilt_deg, 1);
    Serial.print(",");
    Serial.print(s.global_distance, 2);
    Serial.print(" | Speed Step: ");
    Serial.println(s.max_step, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
//  I2C BUS SCAN
// ─────────────────────────────────────────────────────────────────────────────
static bool i2c_check()
{
    bool device_found = false;
    for (byte address = 1; address < 127; ++address)
    {
        Wire.beginTransmission(address);
        if (Wire.endTransmission() == 0)
            device_found = true;
    }
    return device_found;
}