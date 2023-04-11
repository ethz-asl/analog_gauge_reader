import numpy as np


class AngleConverter:
    def __init__(self, theta_zero):
        theta_flipped = np.pi * 2 - theta_zero
        self.theta_zero = theta_flipped

    def convert_angle(self, theta):
        theta_flipped = np.pi * 2 - theta
        theta_shifted = theta_flipped - self.theta_zero
        return theta_shifted
