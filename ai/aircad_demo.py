import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import os
import glob

# --------------------------
# 1. Connect Arduino (Ultrasonic only)
# --------------------------
# Prefer environment variables so you can override without editing the file:
#   ARDUINO_PORT (e.g. /dev/cu.usbserial-110) and ARDUINO_BAUD (e.g. 115200)
ARDUINO_PORT = os.environ.get('ARDUINO_PORT', '/dev/cu.usbserial-110')
ARDUINO_BAUD = int(os.environ.get('ARDUINO_BAUD', '115200'))

def list_serial_ports():
    # list both cu.* and tty.* devices for user convenience
    ports = sorted(glob.glob('/dev/cu.*') + glob.glob('/dev/tty.*'))
    return ports

try:
    ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD)
    time.sleep(2)
except Exception as e:
    print(f"Could not open serial port '{ARDUINO_PORT}' at {ARDUINO_BAUD} baud: {e}")
    print("Available serial devices:")
    for p in list_serial_ports():
        print('  ', p)
    raise

# --------------------------
# 2. Hand Tracking Setup
# --------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Canvas to draw on
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
pen_down = False
thickness = 5  # default thickness

# --------------------------
# Session controls (added)
# --------------------------
# session states: 'idle', 'drawing', 'paused', 'stopped', 'playing'
STATE = 'idle'

strokes = []          # list of strokes; each stroke is list of (x,y,thickness)
current_stroke = None
prev_point = None

def draw_ui(frame, state, distance, thickness, info_msg=""):
    h, w = frame.shape[:2]
    status_text = f"State: {state.upper()}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

    dist_text = f"Distance: {distance if distance is not None and distance > 0 else '--'} cm"
    cv2.putText(frame, dist_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    brush_text = f"Brush: {thickness}"
    cv2.putText(frame, brush_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    lines = [
        "s: start  SPACE: pause/resume  p: play  x: stop  c: clear  q: quit",
        info_msg
    ]
    y = h - 40
    for line in reversed(lines):
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y -= 20

def replay_strokes(strokes, frame_shape, speed=0.002):
    """Blocking replay of recorded strokes. Draws on a fresh canvas and shows animation."""
    if not strokes:
        return
    h, w = frame_shape[:2]
    replay_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    win = "AirSketch - Replay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        for i in range(1, len(stroke)):
            x1, y1, t1 = stroke[i-1]
            x2, y2, t2 = stroke[i]
            cv2.line(replay_canvas, (x1, y1), (x2, y2), (255, 255, 255), int(max(1, (t1+t2)//2)))
            display = cv2.addWeighted(np.zeros_like(replay_canvas), 0.6, replay_canvas, 0.4, 0)
            cv2.imshow(win, display)
            key = cv2.waitKey(1)
            time.sleep(speed)  # tune speed
    # hold final
    cv2.waitKey(800)
    cv2.destroyWindow(win)

info_msg = "Press 's' to start drawing"

# --------------------------
# 3. Main Loop
# --------------------------
while True:
    # -------- Read ultrasonic sensor for thickness -------
    try:
        line = ser.readline().decode().strip()  # Example: "5,1"
        if line != "":
            distance = int(line.split(',')[0])  # Take first value only
        else:
            distance = 20
    except:
        distance = 20

    # Map distance → thickness (closer → thicker)
    thickness = max(1, min(25, int(50 / (distance + 1))))

    # -------- Read webcam --------
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror horizontally for natural movement
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # -------- Hand tracking --------
    if result.multi_hand_landmarks:
        # Use first detected hand as drawing hand
        landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Index finger tip (landmark 8)
        h, w, _ = frame.shape
        x = int(landmarks.landmark[8].x * w)
        y = int(landmarks.landmark[8].y * h)

        # pen_down now depends on STATE: only consider drawing when STATE == 'drawing'
        pen_down = True

        # -- Drawing & recording logic integrated with STATE --
        if STATE == 'drawing':
            if prev_point is None:
                prev_point = (x, y)
                # start a new stroke if needed
                if current_stroke is None:
                    current_stroke = []
            # draw on canvas
            cv2.line(canvas, prev_point, (x, y), (255, 255, 255), thickness)
            # record point with its thickness
            if current_stroke is None:
                current_stroke = []
            current_stroke.append((x, y, thickness))
            prev_point = (x, y)
        else:
            # Not in 'drawing' state: if we had a current stroke and finger is present but not drawing,
            # keep prev_point so drawing can resume correctly only if STATE returns to drawing.
            # However, if we want to finalize stroke when finger lost, we will do that when no hands detected.
            prev_x, prev_y = x, y

    else:
        # no hand detected
        pen_down = False
        if current_stroke is not None and len(current_stroke) > 0:
            strokes.append(current_stroke)
        current_stroke = None
        prev_point = None
        prev_x, prev_y = 0, 0

    combined = cv2.addWeighted(frame, 0.5, canvas, 1, 0)

    draw_ui(combined, STATE, distance, thickness, info_msg)

    cv2.imshow("AirCad Dual-Hand", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        STATE = 'drawing'
        info_msg = "Drawing started"
    elif key == ord(' '):
        if STATE == 'drawing':
            STATE = 'paused'
            info_msg = "Paused"
        elif STATE == 'paused':
            STATE = 'drawing'
            info_msg = "Resumed"
        else:
            info_msg = "No active drawing to pause/resume"
    elif key == ord('x'):
        if current_stroke is not None and len(current_stroke) > 0:
            strokes.append(current_stroke)
        current_stroke = None
        prev_point = None
        STATE = 'stopped'
        info_msg = "Stopped"
    elif key == ord('c'):
        canvas[:] = 0
        strokes = []
        current_stroke = None
        prev_point = None
        STATE = 'idle'
        info_msg = "Cleared"
    elif key == ord('p'):
        if strokes:
            STATE = 'playing'
            info_msg = "Replaying..."
            replay_strokes(strokes, frame.shape)
            info_msg = "Replay finished"
            STATE = 'idle'
        else:
            info_msg = "No strokes to replay"

cap.release()
ser.close()
cv2.destroyAllWindows()
