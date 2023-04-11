import argparse
import cv2
import numpy as np

from plots import Plotter
from gauge_detection.detection_inference import detection_gauge_face
from ocr.ocr_inference import ocr
from key_point_detection.key_point_inference import KeyPointInference, detect_key_points
from geometry.ellipse import fit_ellipse, cart_to_pol, get_polar_angle
from segmentation.segmenation_inference import segment_gauge_needle, \
    get_fitted_line


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help="Path to input image")
    parser.add_argument('--detection_model',
                        type=str,
                        required=True,
                        help="Path to detection model")
    parser.add_argument('--key_point_model',
                        type=str,
                        required=True,
                        help="Path to key point model")
    parser.add_argument('--segmentation_model',
                        type=str,
                        required=True,
                        help="Path to segmentation model")
    parser.add_argument('--base_path',
                        type=str,
                        required=True,
                        help="Path where run folder is stored")
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def crop_image(img, box):
    """
    crop image
    :param img: orignal image
    :param box: in the xyxy format
    :return: cropped image
    """
    cropped_img = img[box[1]:box[3],
                      box[0]:box[2], :]  # image has format [y, x, rgb]
    return cropped_img


def process_image(img_path, detection_model_path, key_point_model,
                  segmentation_model, base_path, debug):
    image = cv2.imread(img_path)

    if debug:
        plotter = Plotter(base_path, image)

        print("-------------------")
        print("Gauge Detection")

    # Gauge detection
    box, all_boxes = detection_gauge_face(image, detection_model_path)

    if debug:
        plotter.plot_bounding_box_img(all_boxes)

    # crop image to only gauge face
    cropped_img = crop_image(image, box)

    # resize
    cropped_img = cv2.resize(cropped_img,
                             dsize=(224, 224),
                             interpolation=cv2.INTER_LINEAR)

    if debug:
        plotter.set_image(cropped_img)
        plotter.plot_image('cropped')

        print("-------------------")
        print("OCR")

    # ocr
    ocr_readings = ocr(cropped_img, debug)

    if debug:
        plotter.plot_ocr(ocr_readings, title='full')

    # get list of ocr readings that are the numbers
    number_labels = []
    for reading in ocr_readings:
        if reading.is_number():
            number_labels.append(reading)

    if debug:
        plotter.plot_ocr(number_labels, title='numbers')

        print("-------------------")
        print("Segmentation")

    needle_mask_x, needle_mask_y = segment_gauge_needle(
        cropped_img, segmentation_model)
    needle_line_coeffs = get_fitted_line(needle_mask_x, needle_mask_y)

    if debug:
        plotter.plot_segmented_line(needle_mask_x, needle_mask_y,
                                    needle_line_coeffs)

        print("-------------------")
        print("Key Point Detection")

    # detect key points
    key_point_inferencer = KeyPointInference(key_point_model)
    heatmaps = key_point_inferencer.predict_heatmaps(cropped_img)
    key_point_list = detect_key_points(heatmaps)

    if debug:
        plotter.plot_heatmaps(heatmaps)
        plotter.plot_key_points(key_point_list)

        print("-------------------")
        print("Ellipse Fitting")

    # fit ellipse to extracted key points
    all_key_points = np.vstack(key_point_list)
    coeffs = fit_ellipse(all_key_points[:, 0], all_key_points[:, 1])
    ellipse_params = cart_to_pol(coeffs)

    if debug:
        plotter.plot_ellipse(all_key_points[:, 0], all_key_points[:, 1],
                             ellipse_params)

        print("-------------------")
        print("Projection")

    for number in number_labels:
        theta = get_polar_angle(number.center, ellipse_params)
        number.set_theta(theta)

    if debug:
        plotter.plot_project_points_ellipse(number_labels, ellipse_params)


def main():
    args = read_args()

    img_path = args.input
    detection_model_path = args.detection_model
    key_point_model = args.key_point_model
    segmentation_model = args.segmentation_model
    base_path = args.base_path

    process_image(img_path,
                  detection_model_path,
                  key_point_model,
                  segmentation_model,
                  base_path,
                  debug=args.debug)


if __name__ == "__main__":
    main()
