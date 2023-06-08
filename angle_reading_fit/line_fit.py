import numpy as np
from sklearn import linear_model


def line_fit(X, y):
    reading_line_coeff = np.polyfit(X, y, 1)
    return reading_line_coeff


def line_fit_ransac(X, y):
    X = X.reshape(-1, 1)
    Y = y.reshape(-1, 1)

    # Fit a line using RANSAC
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, Y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Get the line coefficients
    slope = ransac.estimator_.coef_[0][0]
    intercept = ransac.estimator_.intercept_[0]

    return (slope, intercept), inlier_mask, outlier_mask
