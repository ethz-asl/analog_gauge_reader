import argparse
import cv2

from gauge_detection.detection_inference import detection_gauge_face
from ocr.ocr_inference import ocr


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
    parser.add_argument('--debug',
                        type=bool,
                        required=False,
                        default=False,
                        help="Path to input image")
    return parser.parse_args()


# box xyxy format
def crop_image(img, box):
    cropped_img = img[box[1]:box[3],
                      box[0]:box[2], :]  # image has format [y, x, rgb]
    return cropped_img


# plot detected bounding boxes. boxes is the result of the yolov8 detection
def plot_bounding_box_img(img, boxes):
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
    return img


def plot_img(img, title='image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


def process_image(img_path, detection_model_path, debug=False):
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

    return cropped_img


def main():
    args = read_args()

    img_path = args.input
    detection_model_path = args.detection_model

    process_image(img_path, detection_model_path, debug=args.debug)


if __name__ == "__main__":
    main()
