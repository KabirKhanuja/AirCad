/*
  AirSketch_Ultrasonic.ino
  Simple, robust ultrasonic distance reader for AirSketch project.

  - Uses HC-SR04 (or compatible) ultrasonic sensor
  - Sends an integer distance in centimeters over Serial as a single line
  - Includes simple moving-average smoothing
  - Default pins: TRIG = 9, ECHO = 10
  - Default baud: 115200

  Usage:
  - Upload to your Arduino.
  - Note the serial port (COMx or /dev/ttyUSBx) and use same baud in Python.
  - The Arduino will output lines like:
      23
      24
      24
*/

const uint8_t TRIG_PIN = 9;
const uint8_t ECHO_PIN = 10;

const unsigned long BAUD_RATE = 115200;

// Measurement settings
const unsigned long MEASURE_INTERVAL_MS = 50; // how often to measure
const unsigned int MAX_DISTANCE_CM = 350;     // HC-SR04 practical max ~400 cm
const unsigned long TIMEOUT_US = 30000UL;    // 30 ms timeout (~5 meters) for pulseIn

// Smoothing (moving average)
const uint8_t SMOOTHING_WINDOW = 5;
int readings[SMOOTHING_WINDOW];
uint8_t readIndex = 0;
long total = 0;
uint8_t count = 0;

unsigned long lastMeasure = 0;

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  Serial.begin(BAUD_RATE);
  // Small delay to allow serial monitor / host to open
  delay(200);
  // Initialize smoothing buffer
  for (uint8_t i = 0; i < SMOOTHING_WINDOW; ++i) {
    readings[i] = 0;
  }
}

long measureDistanceCm() {
  // Sends trigger pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Reads echo pulse duration (in microseconds)
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, TIMEOUT_US);

  if (duration == 0) {
    // no echo received (out of range or timeout)
    return 0; // using 0 as "no reading" sentinel
  }

  // speed of sound ~ 343 m/s => 0.0343 cm/us; distance = duration * 0.0343 / 2
  long distanceCm = (long)(duration * 0.0343 / 2.0);

  if (distanceCm < 0) distanceCm = 0;
  if (distanceCm > MAX_DISTANCE_CM) distanceCm = MAX_DISTANCE_CM;

  return distanceCm;
}

int getSmoothedDistance(long newReading) {
  total -= readings[readIndex];
  readings[readIndex] = (int)newReading;
  total += readings[readIndex];

  readIndex = (readIndex + 1) % SMOOTHING_WINDOW;
  if (count < SMOOTHING_WINDOW) count++;

  int average = (int)(total / count);
  return average;
}

void loop() {
  unsigned long now = millis();
  if (now - lastMeasure < MEASURE_INTERVAL_MS) {
    return;
  }
  lastMeasure = now;

  long dist = measureDistanceCm();
  int output = getSmoothedDistance(dist);

  Serial.println(output);
}