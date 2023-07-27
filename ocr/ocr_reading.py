import numpy as np

UNIT_LIST = ["bar", "mbar", "millibars", "MPa", "psi", "C", "°C", "%"]


class OCRReading:
    def __init__(self, polygon, reading, confidence):
        self.polygon = polygon
        self.reading = reading
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
        return self.reading in UNIT_LIST

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
