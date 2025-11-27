import time
import numpy as np
import serial


class DistanceSensorReader:
    """
    Reads distance values (in cm) from an Arduino over Serial.
    The Arduino should send one integer per line using Serial.println().
    """

    def __init__(self, port: str, baud_rate: int = 9600, timeout: float = 1.0):
        """
        :param port: Serial port string, e.g. "COM3" (Windows) or "/dev/tty.usbmodemXXXX" (Mac).
        :param baud_rate: Must match Serial.begin() in Arduino sketch.
        :param timeout: Serial read timeout in seconds.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.last_value = None

        self._connect()

    def _connect(self):
        self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
        # Give Arduino time to reset after opening serial
        time.sleep(2)

    def read_distance(self):
        """
        Returns:
            int or None: distance in cm if a valid value was read, otherwise None.
        """
        if self.ser is None:
            return None

        if self.ser.in_waiting == 0:
            return None

        try:
            line = self.ser.readline().decode(errors="ignore").strip()
            if not line:
                return None

            value = int(float(line))
            self.last_value = value
            return value
        except (ValueError, UnicodeDecodeError):
            # Corrupt / partial line; ignore
            return None

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()


def map_distance_to_thickness(distance_cm,
                              min_dist=5,
                              max_dist=50,
                              min_thickness=2,
                              max_thickness=25):
    """
    Maps ultrasonic distance to brush thickness.

    - Closer hand  -> thicker brush
    - Farther hand -> thinner brush

    If distance_cm is None or invalid, returns a mid value.
    """
    if distance_cm is None or distance_cm <= 0:
        return int((min_thickness + max_thickness) / 2)

    # Clamp distance into [min_dist, max_dist]
    d = max(min_dist, min(max_dist, distance_cm))

    # Invert mapping: small distance -> large thickness
    thickness = np.interp(d,
                          [min_dist, max_dist],
                          [max_thickness, min_thickness])

    return int(thickness)