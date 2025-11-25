# AirCad

AirCad is a gesture-based virtual drawing system that lets users draw in the air using their hand. It combines AI-based hand tracking with Arduino ultrasonic sensing to create a real-time sketching experience.

## Overview
The system uses:
- A webcam and MediaPipe for detecting hand landmarks.
- An Arduino Uno with an ultrasonic sensor to measure hand distance.
- Python (OpenCV + PySerial) for drawing on a virtual canvas and integrating sensor data.

## Features
- Draw in air using your index finger.
- Brush thickness (or color) changes based on hand distance.
- Real-time AI hand tracking.
- Clear and save canvas options.
- Simple Arduino–Python serial communication.

## Hardware Components
| Component | Quantity | Purpose |
|------------|-----------|----------|
| Arduino Uno | 1 | Microcontroller for sensor data |
| Ultrasonic Sensor (HC-SR04) | 1 | Measures hand distance |
| USB Cable | 1 | Arduino connection |
| Webcam | 1 | Captures hand movement |

## Software Requirements
- Arduino IDE
- Python 3.x
- Libraries: `opencv-python`, `mediapipe`, `pyserial`, `numpy`

## Setup

### 1. Arduino
1. Open `arduino/AirSketch_Ultrasonic.ino` in Arduino IDE.
2. Upload it to the board.
3. Note the COM port (e.g., `COM3` or `/dev/ttyUSB0`).

### 2. Python
```bash
cd ai
pip install -r requirements.txt
python airsketch_main.py
