import argparse
import cv2
import matplotlib.pyplot as plt

from gauge_detection.detection_inference import detection_gauge_face
from ocr.ocr_inference import ocr
from key_point_detection.key_point_inference import KeyPointInference


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
def plot_bounding_box_img(img, boxes):
    """
    plot detected bounding boxes. boxes is the result of the yolov8 detection
    :param img: image to draw bounding boxes on
    :param boxes: list of bounding boxes
    """
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

    image = image.permute(1, 2, 0)

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

    # ocr
    ocr_results = ocr(cropped_img, debug)

    if debug:
        plot_img(ocr_results['visualization'][0])

    key_point_inferencer = KeyPointInference(key_point_model)
    key_point_list = key_point_inferencer.detect_key_points(cropped_img)

    if debug:
        plot_key_points(cropped_img, key_point_list)

    return cropped_img


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
