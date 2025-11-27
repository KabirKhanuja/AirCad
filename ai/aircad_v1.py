import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# --------------------------
# 1. Connect Arduino (Ultrasonic only)
# --------------------------
ser = serial.Serial('/dev/tty.usbserial-10', 115200)  # Replace with your Arduino port
time.sleep(2)

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

        pen_down = True  # Draw whenever hand is detected

        # Draw on canvas
        if prev_x != 0 and prev_y != 0:
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), thickness)

        prev_x, prev_y = x, y
    else:
        pen_down = False
        prev_x, prev_y = 0, 0

    # -------- Merge webcam + canvas --------
    combined = cv2.addWeighted(frame, 0.5, canvas, 1, 0)

    # -------- Display info --------
    cv2.putText(combined, f"Thickness: {thickness}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, f"Drawing: {'Yes' if pen_down else 'No'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, f"Distance: {distance} cm", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AirCad Dual-Hand", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------- Cleanup --------
cap.release()
ser.close()
cv2.destroyAllWindows()
