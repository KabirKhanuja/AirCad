# airsketch_main.py
# AirSketch main with session controls and stroke recording
import cv2
import numpy as np
import time
from hand_tracking import HandTracker
from utils import DistanceSensorReader, map_distance_to_thickness

# CONFIG - replace with your detected port
ARDUINO_PORT = "/dev/cu.usbserial-110"  # <--- set your port
BAUD_RATE = 9600

# session states: 'idle', 'drawing', 'paused', 'stopped', 'playing'
STATE = 'idle'

def draw_ui(frame, state, distance, thickness, info_msg=""):
    h, w = frame.shape[:2]
    # top-left info
    status_text = f"State: {state.upper()}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)

    dist_text = f"Distance: {distance if distance is not None and distance > 0 else '--'} cm"
    cv2.putText(frame, dist_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    brush_text = f"Brush: {thickness}"
    cv2.putText(frame, brush_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # instructions bottom-left
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
            cv2.line(replay_canvas, (x1, y1), (x2, y2), (255, 0, 0), int(max(1, (t1+t2)//2)))
            display = cv2.addWeighted(np.zeros_like(replay_canvas), 0.6, replay_canvas, 0.4, 0)
            cv2.imshow(win, display)
            key = cv2.waitKey(1)
            time.sleep(speed)  # tune speed
    # hold final
    cv2.waitKey(800)
    cv2.destroyWindow(win)

def main():
    global STATE
    # try to connect to Arduino
    try:
        distance_reader = DistanceSensorReader(ARDUINO_PORT, BAUD_RATE)
        print(f"Connected to Arduino on {ARDUINO_PORT}")
    except Exception as e:
        print("Error opening serial port:", e)
        print("Continuing without ultrasonic.")
        distance_reader = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    tracker = HandTracker()
    canvas = None
    strokes = []          # list of strokes; each stroke is list of (x,y,thickness)
    current_stroke = None
    prev_point = None

    info_msg = "Press 's' to start drawing"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            frame = cv2.flip(frame, 1)
            if canvas is None:
                canvas = np.zeros_like(frame)

            # read distance (if possible)
            distance = None
            if distance_reader is not None:
                distance = distance_reader.read_distance()

            brush_thickness = map_distance_to_thickness(distance)

            # hand tracking
            fingertip, annotated_frame = tracker.process(frame, draw_landmarks=True)

            # handle drawing logic based on STATE
            if STATE == 'drawing':
                # when drawing: if fingertip present, draw and record
                if fingertip is not None:
                    x, y = fingertip
                    if current_stroke is None:
                        current_stroke = []
                        prev_point = (x, y)
                    # draw on canvas
                    cv2.line(canvas, prev_point, (x, y), (255, 0, 0), brush_thickness)
                    # record point
                    current_stroke.append((x, y, brush_thickness))
                    prev_point = (x, y)
                else:
                    # finger lost: finalize stroke
                    if current_stroke is not None and len(current_stroke) > 0:
                        strokes.append(current_stroke)
                    current_stroke = None
                    prev_point = None

            elif STATE == 'paused':
                # do not draw; keep showing landmarks; if finger lost finalize stroke similarly
                if current_stroke is not None and fingertip is None:
                    if len(current_stroke) > 0:
                        strokes.append(current_stroke)
                    current_stroke = None
                    prev_point = None

            elif STATE == 'idle':
                # not recording or drawing
                pass

            elif STATE == 'stopped':
                # stopped: don't draw; keep canvas as-is (or you may have cleared it)
                pass

            elif STATE == 'playing':
                # playing handled via blocking replay_strokes below, set STATE back to idle afterwards
                pass

            # overlay canvas
            combined = cv2.addWeighted(annotated_frame, 0.6, canvas, 0.4, 0)

            # show UI info
            draw_ui(combined, STATE, distance, brush_thickness, info_msg)

            cv2.imshow("AirSketch", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # start drawing session
                STATE = 'drawing'
                info_msg = "Drawing started"
            elif key == ord(' '):
                # pause/resume toggle
                if STATE == 'drawing':
                    STATE = 'paused'
                    info_msg = "Paused"
                elif STATE == 'paused':
                    STATE = 'drawing'
                    info_msg = "Resumed"
                else:
                    info_msg = "No active drawing to pause/resume"
            elif key == ord('x'):
                # stop: stop drawing and finalize current stroke
                if current_stroke is not None and len(current_stroke) > 0:
                    strokes.append(current_stroke)
                current_stroke = None
                prev_point = None
                STATE = 'stopped'
                info_msg = "Stopped"
            elif key == ord('c'):
                # clear canvas and erase strokes
                canvas[:] = 0
                strokes = []
                current_stroke = None
                prev_point = None
                STATE = 'idle'
                info_msg = "Cleared"
            elif key == ord('p'):
                # play/replay recorded strokes (blocking animation)
                if strokes:
                    STATE = 'playing'
                    info_msg = "Replaying..."
                    # show annotated_frame as background while replaying strokes on blank canvas
                    replay_strokes(strokes, frame.shape)
                    info_msg = "Replay finished"
                    STATE = 'idle'
                else:
                    info_msg = "No strokes to replay"
            # end key handling

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        if distance_reader is not None:
            distance_reader.close()

if __name__ == "__main__":
    main()