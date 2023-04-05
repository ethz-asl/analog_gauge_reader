import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from gauge_detection.detection_inference import detection_gauge_face
from ocr.ocr_inference import ocr
from key_point_detection.key_point_inference import KeyPointInference
from geometry.ellipse import get_ellipse_pts, fit_ellipse, cart_to_pol


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


#
def plot_bounding_box_img(image, boxes):
    """
    plot detected bounding boxes. boxes is the result of the yolov8 detection
    :param img: image to draw bounding boxes on
    :param boxes: list of bounding boxes
    """
    img = np.copy(image)
    for box in boxes:
        bbox = box.xyxy[0].int()
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))

        color_face = (0, 255, 0)
        color_needle = (255, 0, 0)
        if box.cls == 0:
            color = color_face
        else:
            color = color_needle

        img = cv2.rectangle(img,
                            start_point,
                            end_point,
                            color=color,
                            thickness=1)

    plot_img(img)


def plot_key_points(image, key_point_list):
    plt.figure(figsize=(12, 8))

    titles = ['Start', 'Middle', 'End']

    for i in range(3):
        key_points = key_point_list[i]
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.scatter(key_points[:, 0],
                    key_points[:, 1],
                    s=50,
                    c='red',
                    marker='x')
        plt.title(f'Predicted Key Point {titles[i]}')

    plt.tight_layout()

    plt.show()


def plot_img(img, title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


def plot_ellipse(image, x, y, params):
    plt.imshow(image)
    plt.plot(x, y, 'x')  # given points
    x, y = get_ellipse_pts(params)
    plt.plot(x, y)  # given points
    plt.show()


def process_image(img_path,
                  detection_model_path,
                  key_point_model,
                  debug=False):
    image = cv2.imread(img_path)

    # Gauge detection
    box, all_boxes = detection_gauge_face(image, detection_model_path)

    if debug:
        plot_bounding_box_img(image, all_boxes)

    # crop image to only gauge face
    cropped_img = crop_image(image, box)

    # resize
    cropped_img = cv2.resize(cropped_img,
                             dsize=(224, 224),
                             interpolation=cv2.INTER_LINEAR)

    if debug:
        plot_img(cropped_img)

    # ocr
    ocr_results = ocr(cropped_img, debug)

    if debug:
        plot_img(ocr_results['visualization'][0])

    # detect key points
    key_point_inferencer = KeyPointInference(key_point_model)
    key_point_list = key_point_inferencer.detect_key_points(cropped_img, debug)

    if debug:
        plot_key_points(cropped_img, key_point_list)

    all_key_points = np.vstack(key_point_list)

    coeffs = fit_ellipse(all_key_points[:, 0], all_key_points[:, 1])
    x0, y0, ap, bp, phi = cart_to_pol(coeffs)

    if debug:
        plot_ellipse(cropped_img, all_key_points[:, 0], all_key_points[:, 1],
                     (x0, y0, ap, bp, phi))


def main():
    args = read_args()

    img_path = args.input
    detection_model_path = args.detection_model
    key_point_model = args.key_point_model

    process_image(img_path,
                  detection_model_path,
                  key_point_model,
                  debug=args.debug)


if __name__ == "__main__":
    main()
