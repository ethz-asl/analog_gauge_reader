import numpy as np


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

    def set_theta(self, theta):
        self.theta = theta
