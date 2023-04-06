import numpy as np


class OCRReading:
    def __init__(self, polygon, reading, confidence):
        self.polygon = polygon
        self.reading = reading
        self.confidence = confidence

        self.center = self._get_centroid()

    def _get_centroid(self):
        x_mean = np.mean(self.polygon[:, 0])
        y_mean = np.mean(self.polygon[:, 1])

        return (x_mean, y_mean)
