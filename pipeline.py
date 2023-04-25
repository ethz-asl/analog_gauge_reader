import argparse
import cv2
import numpy as np
from PIL import Image

from plots import Plotter
from gauge_detection.detection_inference import detection_gauge_face
from ocr.ocr_inference import ocr
from key_point_detection.key_point_inference import KeyPointInference, detect_key_points
from geometry.ellipse import fit_ellipse, cart_to_pol, get_line_ellipse_point, get_polar_angle
from geometry.angle_converter import AngleConverter
from segmentation.segmenation_inference import get_start_end_line, segment_gauge_needle, \
    get_fitted_line

OCR_THRESHOLD = 0.9


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help=
        "Path to input image. If a directory then it will pass all images of directory"
    )
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
    image = Image.open(img_path).convert("RGB")
    image = np.asarray(image)

    if debug:
        plotter = Plotter(base_path, image)

        print("-------------------")
        print("Gauge Detection")

    # ------------------Gauge detection-------------------------

    box, all_boxes = detection_gauge_face(image, detection_model_path)

    if debug:
        plotter.plot_bounding_box_img(all_boxes)

    # crop image to only gauge face
    cropped_img = crop_image(image, box)

    # resize
    cropped_img = cv2.resize(cropped_img,
                             dsize=(224, 224),
                             interpolation=cv2.INTER_CUBIC)

    if debug:
        plotter.set_image(cropped_img)
        plotter.plot_image('cropped')

        print("-------------------")
        print("OCR")

    # ------------------OCR-------------------------

    ocr_readings, ocr_visualization = ocr(cropped_img, debug)

    if debug:
        plotter.plot_ocr_visualization(ocr_visualization)
        plotter.plot_ocr(ocr_readings, title='full')

    # get list of ocr readings that are the numbers
    number_labels = []
    for reading in ocr_readings:
        if reading.is_number() and reading.confidence > OCR_THRESHOLD:
            number_labels.append(reading)

    if debug:
        plotter.plot_ocr(number_labels, title='numbers')

        print("-------------------")
        print("Segmentation")

    # ------------------Segmentation-------------------------

    needle_mask_x, needle_mask_y = segment_gauge_needle(
        cropped_img, segmentation_model)
    needle_line_coeffs = get_fitted_line(needle_mask_x, needle_mask_y)
    needle_line_start, needle_line_end = get_start_end_line(needle_mask_x)

    if debug:
        plotter.plot_segmented_line(needle_mask_x, needle_mask_y,
                                    needle_line_coeffs)

        print("-------------------")
        print("Key Point Detection")

    # ------------------Key Point Detection-------------------------

    key_point_inferencer = KeyPointInference(key_point_model)
    heatmaps = key_point_inferencer.predict_heatmaps(cropped_img)
    key_point_list = detect_key_points(heatmaps)

    if debug:
        plotter.plot_heatmaps(heatmaps)
        plotter.plot_key_points(key_point_list)

        print("-------------------")
        print("Ellipse Fitting")

    # ------------------Ellipse Fitting-------------------------

    all_key_points = np.vstack(key_point_list)
    coeffs = fit_ellipse(all_key_points[:, 0], all_key_points[:, 1])
    ellipse_params = cart_to_pol(coeffs)

    if debug:
        plotter.plot_ellipse(all_key_points, ellipse_params, 'key_points')

        print("-------------------")
        print("Projection")

    # ------------------Project OCR Numbers to ellipse-------------------------

    if len(number_labels) == 0:
        print("Didn't find any numbers with ocr")
        return

    for number in number_labels:
        theta = get_polar_angle(number.center, ellipse_params)
        if theta < 0:
            theta = 2 * np.pi + theta
        number.set_theta(theta)

    if debug:
        plotter.plot_project_points_ellipse(number_labels, ellipse_params)

    # ------------------Project Needle to ellipse-------------------------

    point_needle_ellipse = get_line_ellipse_point(
        needle_line_coeffs, (needle_line_start, needle_line_end),
        ellipse_params)

    if debug:
        plotter.plot_ellipse(point_needle_ellipse.reshape(1, 2),
                             ellipse_params, 'needle point')


# ------------------Fit line to angles and get reading of needle-------------------------

    needle_angle = get_polar_angle(point_needle_ellipse, ellipse_params)
    if needle_angle < 0:
        needle_angle = 2 * np.pi + needle_angle

    min_number = number_labels[0]
    for number in number_labels:
        if number.number < min_number.number:
            min_number = number

    if debug:
        print(f"Minimum detected number is: {min_number.number}")
    angle_converter = AngleConverter(min_number.theta)

    angle_number_list = []
    for number in number_labels:
        angle_number_list.append(
            (angle_converter.convert_angle(number.theta), number.number))

    angle_number_arr = np.array(angle_number_list)
    reading_line_coeff = np.polyfit(angle_number_arr[:, 0],
                                    angle_number_arr[:, 1], 1)
    reading_line = np.poly1d(reading_line_coeff)

    needle_angle_conv = angle_converter.convert_angle(needle_angle)

    reading = reading_line(needle_angle_conv)

    if debug:
        print(f"Final reading is: {reading}")
        plotter.plot_final_reading_ellipse([], point_needle_ellipse,
                                           round(reading, 1), ellipse_params)


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
