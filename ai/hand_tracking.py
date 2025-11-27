import cv2
import mediapipe as mp


class HandTracker:
    """
    Wrapper around MediaPipe Hands.
    Provides fingertip (index finger) coordinates in image space.
    """

    def __init__(self,
                 max_num_hands=1,
                 detection_confidence=0.7,
                 tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame, draw_landmarks=True):
        """
        Processes a BGR frame and returns:
            (x, y), annotated_frame

        - (x, y): index finger tip coordinates in pixel space, or None if not found.
        - annotated_frame: original frame with landmarks drawn (optional).
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        fingertip = None

        if results.multi_hand_landmarks:
            # Take only the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            # Index finger tip landmark id = 8
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            fingertip = (x, y)

            if draw_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return fingertip, frame

    def close(self):
        self.hands.close()