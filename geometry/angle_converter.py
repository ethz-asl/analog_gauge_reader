import numpy as np


class AngleConverter:
    def __init__(self, theta_zero):
        self.theta_zero = theta_zero

    def convert_angle(self, theta):
        theta_shifted = theta - self.theta_zero
        if theta_shifted < 0:
            theta_shifted = theta_shifted + 2 * np.pi
        return theta_shifted
