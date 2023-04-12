from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    needle_mask = results[0].masks.data[0].numpy()
    needle_mask_resized = cv2.resize(needle_mask,
                                     dsize=(image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

    y_coords, x_coords = np.where(needle_mask_resized)

    return x_coords, y_coords


def get_fitted_line(x_coords, y_coords):
    line_coeffs = np.polyfit(x_coords, y_coords, 1)
    return line_coeffs


def plot_segmented_line(image, x_coords, y_coords, line_coeffs):
    line_fn = np.poly1d(line_coeffs)
    # Plot the line on top of the image
    plt.imshow(image)
    plt.scatter(x_coords, y_coords)
    plt.plot(x_coords, line_fn(x_coords), color='red')
    plt.show()
