import argparse
import json
import os
import sys
import cv2

import numpy as np
from PIL import Image

import constants
from eval_plots import EvalPlotter

# Append path of parent directory to system to import all modules correctly
parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

# pylint: disable=wrong-import-position
from pipeline import crop_image, RESOLUTION
from key_point_detection.key_point_extraction import key_point_metrics

IOU_THRESHOLD = 0.5


# label-studio annotations are always in scale 0,100 so need to rescale
def convert_bbox_annotation(single_bbox_dict, img_width, img_height):
    single_bbox_dict['x'] *= img_width / 100
    single_bbox_dict['y'] *= img_height / 100
    single_bbox_dict['width'] *= img_width / 100
    single_bbox_dict['height'] *= img_height / 100


# label-studio annotations are always in scale 0,100 so need to rescale
def convert_keypoint_annotation(single_bbox_dict, img_width, img_height):
    single_bbox_dict['x'] *= img_width / 100
    single_bbox_dict['y'] *= img_height / 100


def get_annotations_bbox(data):
    """
    Function to extract annotation from format of label-studio
    """

    annotation_dict = {}

    for data_point in data:

        bbox_annotations = {}
        # Get image name. We have image name in format :
        # /data/upload/1/222ae49e-1_cropped_000001_jpg.rf.c7410b0b01b2bc3a6cdff656618a3015.jpg
        # get rid of everything before the '-'
        idx = data_point['data']['image'].find('-') + 1
        image_name = data_point['data']['image'][idx:]

        img_width = data_point['annotations'][0]['result'][0]['original_width']
        img_height = data_point['annotations'][0]['result'][0][
            'original_height']

        bbox_annotations[constants.IMG_SIZE_KEY] = {
            'width': img_width,
            'height': img_height
        }

        bbox_annotations[constants.OCR_NUM_KEY] = []

        for annotation in data_point['annotations'][0]['result']:

            # Image original size saved for each annotation individually.
            # check that all are the same
            assert img_width == annotation['original_width'] and \
                    img_height == annotation['original_height']

            single_bbox_dict = {
                k: annotation['value'][k]
                for k in ('x', 'y', 'width', 'height')
            }
            convert_bbox_annotation(single_bbox_dict, img_width, img_height)

            if annotation['value']['rectanglelabels'][
                    0] == constants.OCR_NUM_KEY:
                bbox_annotations[constants.OCR_NUM_KEY].append(
                    single_bbox_dict)

            if annotation['value']['rectanglelabels'][
                    0] == constants.OCR_UNIT_KEY:
                bbox_annotations[constants.OCR_UNIT_KEY] = single_bbox_dict

            if annotation['value']['rectanglelabels'][
                    0] == constants.GAUGE_DET_KEY:
                bbox_annotations[constants.GAUGE_DET_KEY] = single_bbox_dict

        annotation_dict[image_name] = bbox_annotations

    return annotation_dict


def get_annotations_keypoint(data):
    """
    Function to extract annotation from format of label-studio
    """

    annotation_dict = {}

    for data_point in data:

        keypoint_annotations = {}
        # Get image name. We have image name in format :
        # /data/upload/1/222ae49e-1_cropped_000001_jpg.rf.c7410b0b01b2bc3a6cdff656618a3015.jpg
        # get rid of everything before the '-'
        idx = data_point['data']['img'].find('-') + 1
        image_name = data_point['data']['img'][idx:]

        img_width = data_point['annotations'][0]['result'][0]['original_width']
        img_height = data_point['annotations'][0]['result'][0][
            'original_height']

        keypoint_annotations[constants.IMG_SIZE_KEY] = {
            'width': img_width,
            'height': img_height
        }

        keypoint_annotations[constants.KEYPOINT_NOTCH_KEY] = []

        for annotation in data_point['annotations'][0]['result']:

            # Image original size saved for each annotation individually.
            # check that all are the same
            assert img_width == annotation['original_width'] and \
                    img_height == annotation['original_height']

            single_keypoint_dict = {
                k: annotation['value'][k]
                for k in ('x', 'y')
            }
            convert_keypoint_annotation(single_keypoint_dict, img_width,
                                        img_height)
            keypoint_annotations[constants.KEYPOINT_NOTCH_KEY].append(
                single_keypoint_dict)

            if annotation['value']['keypointlabels'][
                    0] == constants.KEYPOINT_START_KEY:
                keypoint_annotations[
                    constants.KEYPOINT_START_KEY] = single_keypoint_dict

            if annotation['value']['keypointlabels'][
                    0] == constants.KEYPOINT_END_KEY:
                keypoint_annotations[
                    constants.KEYPOINT_END_KEY] = single_keypoint_dict

        annotation_dict[image_name] = keypoint_annotations

    return annotation_dict


def get_annotations_from_json(bbox_path, key_point_path):
    """
    returns annotation dict with each image name as a key.
    For each we have another dict, with a key for each result of the different stages.
    """
    with open(bbox_path, 'r') as file:
        bbox_true_dict = json.load(file)
    with open(key_point_path, 'r') as file:
        keypoint_true_dict = json.load(file)

    bbox_dict = get_annotations_bbox(bbox_true_dict)
    key_point_dict = get_annotations_keypoint(keypoint_true_dict)

    assert set(bbox_dict.keys()) == set(key_point_dict.keys())

    full_annotations = {}
    for key in bbox_dict:

        bbox_img_width = bbox_dict[key][constants.IMG_SIZE_KEY]['width']
        bbox_img_height = bbox_dict[key][constants.IMG_SIZE_KEY]['height']
        keypoint_img_width = key_point_dict[key][
            constants.IMG_SIZE_KEY]['width']
        keypoint_img_height = key_point_dict[key][
            constants.IMG_SIZE_KEY]['height']
        assert bbox_img_width == keypoint_img_width  and \
                bbox_img_height == keypoint_img_height

        full_annotations[key] = {
            constants.IMG_SIZE_KEY:
            bbox_dict[key][constants.IMG_SIZE_KEY],
            constants.OCR_NUM_KEY:
            bbox_dict[key][constants.OCR_NUM_KEY],
            constants.OCR_UNIT_KEY:
            bbox_dict[key][constants.OCR_UNIT_KEY],
            constants.GAUGE_DET_KEY:
            bbox_dict[key][constants.GAUGE_DET_KEY],
            constants.KEYPOINT_NOTCH_KEY:
            key_point_dict[key][constants.KEYPOINT_NOTCH_KEY],
            constants.KEYPOINT_START_KEY:
            key_point_dict[key][constants.KEYPOINT_START_KEY],
            constants.KEYPOINT_END_KEY:
            key_point_dict[key][constants.KEYPOINT_END_KEY],
        }

    return full_annotations


def write_json(path, dictionary):
    result = json.dumps(dictionary, indent=4)
    with open(path, "w") as outfile:
        outfile.write(result)


def get_predictions(run_path):
    prediction_results = {}

    for subdir in os.listdir(run_path):

        subdirectory = os.path.join(run_path, subdir)
        if os.path.isdir(subdirectory):

            result_file = os.path.join(subdirectory,
                                       constants.RESULT_FULL_FILE_NAME)
            if os.path.isfile(result_file):
                with open(result_file, 'r') as file:
                    result_dict = json.load(file)
                    result_dict[constants.ORIGINAL_IMG_KEY] = os.path.join(
                        subdirectory, constants.ORIGINAL_IMG_FILE_NAME)
                    prediction_results[subdir] = result_dict

            else:
                prediction_results[subdir] = constants.FAILED
                print("Error: No Prediction file found! \
                        Pipeline failed before prediction could be made")

    outfile_path = os.path.join(run_path, "predictions_full_results.json")
    predictions_json = json.dumps(prediction_results, indent=4)
    with open(outfile_path, "w") as outfile:
        outfile.write(predictions_json)

    return prediction_results


def bb_intersection_over_union(boxA, boxB):
    """
    boxA and boxB here are of format (x,y, width, height)
    adapted from
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # coordinates of the area of intersection.
    xA = max(boxA['x'], boxB['x'])
    yA = max(boxA['y'], boxB['y'])
    xB = min(boxA['x'] + boxA['width'], boxB['x'] + boxB['width'])
    yB = min(boxA['y'] + boxA['height'], boxB['y'] + boxB['height'])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA['width'] + 1) * (boxA['height'] + 1)
    boxBArea = (boxB['width'] + 1) * (boxB['height'] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compare_gauge_detecions(annotation, prediction, plotter, eval_dict):
    plotter.plot_bounding_box_img([annotation], [prediction],
                                  'gauge detection')
    iou = bb_intersection_over_union(annotation, prediction)
    eval_dict[constants.GAUGE_IOU_KEY] = iou


def compare_ocr_numbers(annotation, prediction, plotter, eval_dict):
    plotter.plot_bounding_box_img(annotation, prediction,
                                  'scale marker detection')

    n_ocr_detected = 0
    for annotation_bbox in annotation:
        iou_max = 0
        for prediction_bbox in prediction:
            iou = bb_intersection_over_union(annotation_bbox, prediction_bbox)
            iou_max = max(iou, iou_max)
        if iou_max > IOU_THRESHOLD:
            n_ocr_detected += 1
    eval_dict[constants.N_OCR_DETECTED_KEY] = n_ocr_detected


def compare_notches(annotation_list, prediction_list, plotter, eval_dict):
    #bring points to right format for key point metrics function.
    #need 2d array for this
    annotation = []
    for annotation_dict in annotation_list:
        annotation.append([annotation_dict['x'], annotation_dict['y']])
    predicted = []
    for prediction_dict in prediction_list:
        predicted.append([prediction_dict['x'], prediction_dict['y']])
    annotation = np.array(annotation)
    predicted = np.array(predicted)

    # plot keypoints
    plotter.plot_key_points(annotation, predicted, 'notches')

    metrics_dict = key_point_metrics(predicted, annotation)
    eval_dict[constants.NOTCHES_METRICS_KEY] = metrics_dict


def compare_single_keypoint(annotation, prediction, plotter, eval_dict,
                            is_start):
    """
    this is for start and end notch evaluation. if is_start then start, else end
    """

    #bring points to right format for key point metrics function.
    #need 2d array for this
    annotation = np.array([[annotation['x'], annotation['y']]])
    predicted = np.array([[prediction['x'], prediction['y']]])

    metrics_dict = key_point_metrics(predicted, annotation)

    if is_start:
        plotter.plot_key_points(annotation, predicted, 'start notch')
        eval_dict[constants.START_METRICS_KEY] = metrics_dict
    else:
        plotter.plot_key_points(annotation, predicted, 'end notch')
        eval_dict[constants.END_METRICS_KEY] = metrics_dict


def is_point_inside(point, crop_box):
    return point['x']>crop_box['x'] and point['x']<crop_box['x']+crop_box['width'] \
        and point['y']>crop_box['y'] and point['y']<crop_box['y']+crop_box['height']


def is_bbox_inside(bbox, crop_box):
    point1 = {'x': bbox['x'], 'y': bbox['y']}
    point2 = {'x': bbox['x'] + bbox['width'], 'y': bbox['y'] + bbox['height']}
    return is_point_inside(point1, crop_box) and is_point_inside(
        point2, crop_box)


def rescale_point(point, crop_box, border):
    if is_point_inside(point, crop_box):
        top, bottom, left, right = border

        x_offset = crop_box['x'] - left
        y_offset = crop_box['y'] - top

        box_width = crop_box['width'] + left + right
        box_height = crop_box['height'] + top + bottom
        rescale_resolution = RESOLUTION

        x_shift = point['x'] - x_offset
        y_shift = point['y'] - y_offset

        point['x'] = x_shift * rescale_resolution[0] / box_width
        point['y'] = y_shift * rescale_resolution[1] / box_height


def rescale_bbox(bbox, crop_box, border):
    if is_bbox_inside(bbox, crop_box):
        top, bottom, left, right = border

        x_offset = crop_box['x'] - left
        y_offset = crop_box['y'] - top

        box_width = crop_box['width'] + left + right
        box_height = crop_box['height'] + top + bottom
        rescale_resolution = RESOLUTION

        x_shift = bbox['x'] - x_offset
        y_shift = bbox['y'] - y_offset

        bbox['x'] = x_shift * rescale_resolution[0] / box_width
        bbox['y'] = y_shift * rescale_resolution[1] / box_height
        bbox['width'] *= rescale_resolution[0] / box_width
        bbox['height'] *= rescale_resolution[0] / box_width


def main(bbox_path, key_point_path, run_path):

    annotations_dict = get_annotations_from_json(bbox_path, key_point_path)
    predictions_dict = get_predictions(run_path)

    assert set(predictions_dict.keys()) == set(annotations_dict.keys())

    full_eval_dict = {}

    for image_name in annotations_dict:

        annotation_dict = annotations_dict[image_name]
        prediction_dict = predictions_dict[image_name]

        # get corresponding image for plots
        image_path = prediction_dict[constants.ORIGINAL_IMG_KEY]
        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image)

        eval_dict = {}

        eval_path = os.path.join(run_path, image_name, "eval")
        os.makedirs(eval_path, exist_ok=True)
        plotter = EvalPlotter(eval_path, image)

        # compare gauge detection
        compare_gauge_detecions(annotation_dict[constants.GAUGE_DET_KEY],
                                prediction_dict[constants.GAUGE_DET_KEY],
                                plotter, eval_dict)

        # Crop and rescale image
        pred_gauge_bbox = prediction_dict[constants.GAUGE_DET_KEY]
        pred_gauge_bbox_list = [
            pred_gauge_bbox['x'], pred_gauge_bbox['y'],
            pred_gauge_bbox['x'] + pred_gauge_bbox['width'],
            pred_gauge_bbox['y'] + pred_gauge_bbox['height']
        ]
        cropped_img, border = crop_image(image, pred_gauge_bbox_list, True)
        # resize
        cropped_img = cv2.resize(cropped_img,
                                 dsize=RESOLUTION,
                                 interpolation=cv2.INTER_CUBIC)
        plotter.set_image(cropped_img)
        plotter.plot_image('cropped')

        # Crop and rescale all annotations
        for bbox in annotation_dict[constants.OCR_NUM_KEY]:
            rescale_bbox(bbox, pred_gauge_bbox, border)
        rescale_bbox(annotation_dict[constants.OCR_UNIT_KEY], pred_gauge_bbox,
                     border)
        rescale_point(annotation_dict[constants.KEYPOINT_START_KEY],
                      pred_gauge_bbox, border)
        rescale_point(annotation_dict[constants.KEYPOINT_END_KEY],
                      pred_gauge_bbox, border)
        for point in annotation_dict[constants.KEYPOINT_NOTCH_KEY]:
            rescale_point(point, pred_gauge_bbox, border)

        # compare OCR number detection
        compare_ocr_numbers(annotation_dict[constants.OCR_NUM_KEY],
                            prediction_dict[constants.OCR_NUM_KEY], plotter,
                            eval_dict)

        # compare key points
        compare_notches(annotation_dict[constants.KEYPOINT_NOTCH_KEY],
                        prediction_dict[constants.KEYPOINT_NOTCH_KEY], plotter,
                        eval_dict)

        compare_single_keypoint(annotation_dict[constants.KEYPOINT_START_KEY],
                                prediction_dict[constants.KEYPOINT_START_KEY],
                                plotter, eval_dict, True)

        compare_single_keypoint(annotation_dict[constants.KEYPOINT_END_KEY],
                                prediction_dict[constants.KEYPOINT_END_KEY],
                                plotter, eval_dict, False)

        # compare OCR unit detection

        # compare needle segmentations

        # maybe compare line fit and ellipse fit

        # Add eval dict to full
        full_eval_dict[image_name] = eval_dict

        # Save eval_dict to image specific folder
        outfile_path = os.path.join(eval_path, "evaluation.json")
        write_json(outfile_path, eval_dict)

    # Save full eval dict to json
    outfile_path = os.path.join(run_path, "full_evaluation.json")
    write_json(outfile_path, full_eval_dict)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_true_path',
                        type=str,
                        required=True,
                        help="Path to json file with labels for bbox")
    parser.add_argument('--keypoint_true_path',
                        type=str,
                        required=True,
                        help="Path to json file with labels for keypoints")
    parser.add_argument('--run_path',
                        type=str,
                        required=True,
                        help="Path to run folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    main(args.bbox_true_path, args.keypoint_true_path, args.run_path)
