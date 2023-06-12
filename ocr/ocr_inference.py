import math

import numpy as np
import cv2
from mmocr.apis import MMOCRInferencer

from ocr.ocr_reading import OCRReading


def ocr(img, visualize=True):
    """
    Detect and recognize the characters in the image
    :param img: numpy img to do ocr on
    :param visualize: bool if to return image with visualization in results dict
    :return: ocr_results_dict with two keys: 'predictions' what we care about
     and 'visualization' the image for debugging/understanding
    """
    ocr_model = MMOCRInferencer(det='DB_r18', rec='ABINet')

    readings = []

    # MMOCR seems to throw error if no text detected
    try:
        results = ocr_model(img, return_vis=visualize)

        visualization = results['visualization'][0]

        polygons = results['predictions'][0]['det_polygons']

        shapes = []
        for coord_list in polygons:
            shape_array = np.array(coord_list)
            shape_array = shape_array.reshape(-1, 2)
            shapes.append(shape_array)

        scores = results['predictions'][0]['rec_scores']
        texts = results['predictions'][0]['rec_texts']

        assert len(scores) == len(texts) and len(scores) == len(shapes)

        for index, score in enumerate(scores):
            reading = OCRReading(shapes[index], texts[index], score)
            readings.append(reading)

    except IndexError:
        print("nothing detected")

    if visualize:
        return readings, visualization

    return readings


def ocr_rotations(img, plotter, debug):
    degree_list = [0, 45, 90, 135, 180, 225, 270, 315]

    max_conf = -1
    max_num_of_numericals = -1
    max_unit_detected = False

    # try different rotations.
    for degree in degree_list:
        rot_img = rotate(img, degree)
        ocr_readings, ocr_visualization = ocr(rot_img, visualize=True)
        if debug:
            plotter.plot_ocr_visualization(ocr_visualization, degree)

        number_of_numericals = 0
        cumulative_confidence = 0
        unit_detected = False
        for ocr_reading in ocr_readings:
            # only consider readings with high confidence
            if ocr_reading.confidence > 0.5:
                cumulative_confidence += ocr_reading.confidence
                if ocr_reading.is_number():
                    number_of_numericals += 1
            if ocr_reading.is_unit():
                unit_detected = True

        # pick the rotation with the most numericals recognized.
        # In the case of a tie and pick the one that recognizes the unit.
        # If neither or both do, pick the one with most overall confidence.
        # pylint: disable-next = too-many-boolean-expressions
        if (number_of_numericals > max_num_of_numericals
                or (number_of_numericals == max_num_of_numericals
                    and unit_detected and not max_unit_detected)
                or (number_of_numericals == max_num_of_numericals
                    and max_unit_detected == unit_detected
                    and cumulative_confidence > max_conf)):

            max_unit_detected = unit_detected
            max_num_of_numericals = number_of_numericals
            max_conf = cumulative_confidence
            best_ocr_readings = ocr_readings
            best_ocr_visualization = ocr_visualization
            best_degree = degree
            best_rot_img = rot_img

    #rotate the ocr reading polygons back to the unrotated image. Rotate each point individually
    height, width = best_rot_img.shape[:2]
    for ocr_reading in best_ocr_readings:
        polygon = ocr_reading.polygon
        new_polygon = []
        for idx in range(len(polygon)):
            point = polygon[idx, :]
            x_rot, y_rot = rotate_point(point[0], point[1], width, height,
                                        best_degree)
            new_polygon.append([x_rot, y_rot])
        ocr_reading.set_polygon(np.array(new_polygon))

    return best_ocr_readings, best_ocr_visualization, best_degree


def rotate(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle,
                                              1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def rotate_point(x, y, image_width, image_height, rotation_angle):
    center_x = image_width / 2
    center_y = image_height / 2

    # Translate the point
    translated_x = x - center_x
    translated_y = y - center_y

    # Rotate the point
    theta = math.radians(rotation_angle)
    rotated_x = translated_x * math.cos(theta) - translated_y * math.sin(theta)
    rotated_y = translated_x * math.sin(theta) + translated_y * math.cos(theta)

    # Translate the point back
    x_rotated = rotated_x + center_x
    y_rotated = rotated_y + center_y

    return x_rotated, y_rotated
