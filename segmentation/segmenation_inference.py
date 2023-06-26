from ultralytics import YOLO
import numpy as np
import cv2
from scipy import odr


def segment_gauge_needle(image, model_path='best.pt'):
    """
    uses fine-tuned yolo v8 to get mask of segmentation
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: segmentation of needle
    """
    model = YOLO(model_path)  # load model

    results = model.predict(
        image)  # run inference, detects gauge face and needle

    # get list of detected boxes, already sorted by confidence
    try:
        needle_mask = results[0].masks.data[0].numpy()
    except:
        needle_mask = results[0].masks.data[0].cpu().numpy()
    needle_mask_resized = cv2.resize(needle_mask,
                                     dsize=(image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

    y_coords, x_coords = np.where(needle_mask_resized)

    return x_coords, y_coords


def get_fitted_line(x_coords, y_coords):
    """
    Do orthogonal distance regression (odr) for this.
    """
    odr_model = odr.Model(linear)
    data = odr.Data(x_coords, y_coords)
    ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[0.2, 1.], maxit=600)
    out = ordinal_distance_reg.run()
    line_coeffs = out.beta
    residual_variance = out.res_var
    return line_coeffs, residual_variance


def linear(B, x):
    return B[0] * x + B[1]


def get_start_end_line(needle_mask):
    return np.min(needle_mask), np.max(needle_mask)


def cut_off_line(x, y_min, y_max, line_coeffs):
    line = np.poly1d(line_coeffs)
    y = line(x)
    _cut_off(x, y, y_min, y_max, line_coeffs, 0)
    _cut_off(x, y, y_min, y_max, line_coeffs, 1)
    return x[0], x[1]


def _cut_off(x, y, y_min, y_max, line_coeffs, i):
    if y[i] > y_max:
        y[i] = y_max
        x[i] = 1 / line_coeffs[0] * (y_max - line_coeffs[1])
    if y[i] < y_min:
        y[i] = y_min
        x[i] = 1 / line_coeffs[0] * (y_min - line_coeffs[1])
