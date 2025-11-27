import cv2
import numpy as np
from hand_tracking import HandTracker
from utils import DistanceSensorReader, map_distance_to_thickness

# ====== CONFIG ======
# Change this to your Arduino serial port.
# On Mac it will look like: "/dev/tty.usbmodemXXXX" or "/dev/tty.usbserialXXXX"
# You can find it in Arduino IDE: Tools -> Port
ARDUINO_PORT = "/dev/tty.usbmodemXXXX"  # TODO: set correct port
BAUD_RATE = 9600
# =====================


def main():
    # Set up distance sensor reader
    try:
        distance_reader = DistanceSensorReader(ARDUINO_PORT, BAUD_RATE)
        print(f"Connected to Arduino on {ARDUINO_PORT}")
    except Exception as e:
        print("Error opening serial port:", e)
        print("You can still test hand tracking without ultrasonic.")
        distance_reader = None

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    # Create hand tracker
    tracker = HandTracker()

    canvas = None
    prev_point = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Mirror the image so movement feels natural
            frame = cv2.flip(frame, 1)

            # Initialize canvas with same size as frame
            if canvas is None:
                canvas = np.zeros_like(frame)

            # 1. Read distance from ultrasonic (if available)
            distance = None
            if distance_reader is not None:
                distance = distance_reader.read_distance()

            brush_thickness = map_distance_to_thickness(distance)

            # 2. Hand tracking: get fingertip position
            fingertip, annotated_frame = tracker.process(frame, draw_landmarks=True)

            # 3. Drawing logic
            if fingertip is not None:
                x, y = fingertip

                if prev_point is None:
                    prev_point = (x, y)

                # Draw line on canvas
                cv2.line(canvas, prev_point, (x, y), (255, 0, 0), brush_thickness)
                prev_point = (x, y)
            else:
                prev_point = None

            # 4. Overlay canvas on camera feed
            combined = cv2.addWeighted(annotated_frame, 0.6, canvas, 0.4, 0)

            # 5. Show distance / brush info
            if distance is not None and distance > 0:
                dist_text = f"Distance: {distance} cm"
            else:
                dist_text = "Distance: -- cm"

            cv2.putText(combined, dist_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(combined, f"Brush: {brush_thickness}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            cv2.putText(combined, "Press 'c' to clear, 'q' to quit",
                        (20, combined.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("AirSketch", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas[:] = 0  # clear canvas

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        if distance_reader is not None:
            distance_reader.close()


if __name__ == "__main__":
    main()