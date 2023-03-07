import cv2

from gauge_detection.detection_inference import detection_gauge_face
from ocr.ocr_inference import ocr


# box xyxy format
def crop_image(img, box):
    cropped_img = img[box[1]:box[3],
                      box[0]:box[2], :]  # image has format [y, x, rgb]
    return cropped_img


def process_image(img_path):
    image = cv2.imread(img_path)

    # Gauge detection
    box = detection_gauge_face(image)

    # crop image to only gauge face
    cropped_img = crop_image(image, box)

    #ocr
    ocr_results = ocr(cropped_img)

    return ocr_results
