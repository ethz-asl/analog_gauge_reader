import argparse
import os
import logging
import time
import cv2
import numpy as np
from PIL import Image

from plots import RUN_PATH, Plotter
from gauge_detection.detection_inference import detection_gauge_face
from ocr.ocr_inference import ocr
from key_point_detection.key_point_inference import KeyPointInference, detect_key_points
from geometry.ellipse import fit_ellipse, cart_to_pol, get_line_ellipse_point, get_polar_angle
from geometry.angle_converter import AngleConverter
from segmentation.segmenation_inference import get_start_end_line, segment_gauge_needle, \
    get_fitted_line

OCR_THRESHOLD = 0.9
RESOLUTION = (
    448, 448
)  # make sure both dimensions are multiples of 14 for keypoint detection


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
    img = np.copy(img)
    cropped_img = img[box[1]:box[3],
                      box[0]:box[2], :]  # image has format [y, x, rgb]

    height = int(box[3] - box[1])
    width = int(box[2] - box[0])

    # want to preserve aspect ratio but make image square, so do padding
    if height > width:
        delta = height - width
        left, right = delta // 2, delta - (delta // 2)
        top = bottom = 0
    else:
        delta = width - height
        top, bottom = delta // 2, delta - (delta // 2)
        left = right = 0

    pad_color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(cropped_img,
                                 top,
                                 bottom,
                                 left,
                                 right,
                                 cv2.BORDER_CONSTANT,
                                 value=pad_color)
    return new_img


def process_image(img_path, detection_model_path, key_point_model,
                  segmentation_model, base_path, debug):

    logging.info("Start processing image at path %s", img_path)

    image = Image.open(img_path).convert("RGB")
    image = np.asarray(image)

    if debug:
        plotter = Plotter(base_path, image)

    # ------------------Gauge detection-------------------------
    if debug:
        print("-------------------")
        print("Gauge Detection")

    logging.info("Start Gauge Detection")

    box, all_boxes = detection_gauge_face(image, detection_model_path)

    if debug:
        plotter.plot_bounding_box_img(all_boxes)

    # crop image to only gauge face
    cropped_img = crop_image(image, box)

    # resize
    cropped_img = cv2.resize(cropped_img,
                             dsize=RESOLUTION,
                             interpolation=cv2.INTER_CUBIC)

    if debug:
        plotter.set_image(cropped_img)
        plotter.plot_image('cropped')

    logging.info("Finish Gauge Detection")

    # ------------------Key Point Detection-------------------------

    if debug:
        print("-------------------")
        print("Key Point Detection")

    logging.info("Start key point detection")

    key_point_inferencer = KeyPointInference(key_point_model)
    heatmaps = key_point_inferencer.predict_heatmaps(cropped_img)
    key_point_list = detect_key_points(heatmaps)

    if debug:
        plotter.plot_heatmaps(heatmaps)
        plotter.plot_key_points(key_point_list)

    logging.info("Finish key point detection")

    # ------------------Ellipse Fitting-------------------------

    if debug:
        print("-------------------")
        print("Ellipse Fitting")

    logging.info("Start ellipse fitting")

    all_key_points = np.vstack(key_point_list)
    coeffs = fit_ellipse(all_key_points[:, 0], all_key_points[:, 1])
    ellipse_params = cart_to_pol(coeffs)

    if debug:
        plotter.plot_ellipse(all_key_points, ellipse_params, 'key_points')

    logging.info("Finish ellipse fitting")

    # ------------------OCR-------------------------

    if debug:
        print("-------------------")
        print("OCR")

    logging.info("Start OCR")

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

    logging.info("Finish OCR")

    # ------------------Segmentation-------------------------

    if debug:
        print("-------------------")
        print("Segmentation")

    logging.info("Start segmentation")

    needle_mask_x, needle_mask_y = segment_gauge_needle(
        cropped_img, segmentation_model)
    needle_line_coeffs = get_fitted_line(needle_mask_x, needle_mask_y)
    needle_line_start, needle_line_end = get_start_end_line(needle_mask_x)

    if debug:
        plotter.plot_segmented_line(needle_mask_x, needle_mask_y,
                                    needle_line_coeffs)

    logging.info("Finish segmentation")

    # ------------------Project OCR Numbers to ellipse-------------------------

    if debug:
        print("-------------------")
        print("Projection")

    logging.info("Do projection on ellipse")

    if len(number_labels) == 0:
        print("Didn't find any numbers with ocr")
        logging.error("Didn't find any numbers with ocr")
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


def write_dict_to_file(filename, params):

    with open(filename, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def main():
    args = read_args()

    input_path = args.input
    detection_model_path = args.detection_model
    key_point_model = args.key_point_model
    segmentation_model = args.segmentation_model
    base_path = args.base_path

    time_str = time.strftime("%Y%m%d%H%M")
    base_path = os.path.join(base_path, RUN_PATH + '_' + time_str)
    os.makedirs(base_path)

    args_dict = vars(args)
    file_path = os.path.join(base_path, "arguments.txt")
    write_dict_to_file(file_path, args_dict)

    log_path = os.path.join(base_path, "run.log")

    logging.basicConfig(filename=log_path,
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    if os.path.isfile(input_path):
        process_image(input_path,
                      detection_model_path,
                      key_point_model,
                      segmentation_model,
                      base_path,
                      debug=args.debug)
    elif os.path.isdir(input_path):
        for image in os.listdir(input_path):
            img_path = os.path.join(input_path, image)
            try:
                process_image(img_path,
                              detection_model_path,
                              key_point_model,
                              segmentation_model,
                              base_path,
                              debug=args.debug)

            # pylint: disable=broad-except
            # For now want to catch general exceptions and still continue with the other images.
            except Exception as err:
                err_msg = f"Unexpected {err=}, {type(err)=}"
                print(err_msg)
                logging.error(err_msg)


if __name__ == "__main__":
    main()
