import numpy as np

UNIT_LIST = ["bar", "mbar", "millibars", "MPa", "psi", "C", "°C", "F", "°F", "%"]


def restore_decimal_after_leading_zero(reading):
    """
    OCR often misses the decimal, so a scale mark like 0.05 comes back as 005.
    If the text is only digits and starts with 0, put the decimal right after
    that first zero: 005 -> 0.05, 01 -> 0.1, 015 -> 0.15, 02 -> 0.2.
    Readings that already contain a decimal are left unchanged.
    """
    if len(reading) > 1 and reading.startswith("0") and reading.isdigit():
        return "0." + reading[1:]
    return reading


class OCRReading:
    def __init__(self, polygon, reading, confidence):
        self.polygon = polygon
        self.reading = restore_decimal_after_leading_zero(reading.strip())
        self.confidence = confidence

        if self.is_number():
            self.number = float(self.reading)

        self.center = self._get_centroid()

        self.theta = None

    def _get_centroid(self):
        x_mean = np.mean(self.polygon[:, 0])
        y_mean = np.mean(self.polygon[:, 1])

        return (x_mean, y_mean)

    def is_number(self):
        try:
            float(self.reading)
            return True
        except ValueError:
            return False

    def is_unit(self):
        return self.reading.lower() in [unit.lower() for unit in UNIT_LIST]

    def set_polygon(self, polygon):
        self.polygon = polygon
        self.center = self._get_centroid()

    def set_theta(self, theta):
        self.theta = theta

    def get_bounding_box(self):
        x_min = np.min(self.polygon[:, 0])
        y_min = np.min(self.polygon[:, 1])
        x_max = np.max(self.polygon[:, 0])
        y_max = np.max(self.polygon[:, 1])

        return (x_min, y_min, x_max, y_max)
