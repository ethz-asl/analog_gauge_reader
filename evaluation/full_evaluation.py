import argparse
import json
import os

import numpy as np
from PIL import Image

import constants
from eval_plots import EvalPlotter


def convert_bbox_annotation(single_bbox_dict, img_width, img_height):
    single_bbox_dict['x'] *= img_width / 100
    single_bbox_dict['y'] *= img_height / 100
    single_bbox_dict['width'] *= img_width / 100
    single_bbox_dict['height'] *= img_height / 100


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
        bbox_annotations[constants.OCR_UNIT_KEY] = []
        bbox_annotations[constants.GAUGE_DET_KEY] = []

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
                bbox_annotations[constants.OCR_UNIT_KEY].append(
                    single_bbox_dict)

            if annotation['value']['rectanglelabels'][
                    0] == constants.GAUGE_DET_KEY:
                bbox_annotations[constants.GAUGE_DET_KEY].append(
                    single_bbox_dict)

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
        keypoint_annotations[constants.KEYPOINT_START_KEY] = []
        keypoint_annotations[constants.KEYPOINT_END_KEY] = []

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
                keypoint_annotations[constants.KEYPOINT_START_KEY].append(
                    single_keypoint_dict)

            if annotation['value']['keypointlabels'][
                    0] == constants.KEYPOINT_END_KEY:
                keypoint_annotations[constants.KEYPOINT_END_KEY].append(
                    single_keypoint_dict)

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
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compare_gauge_detecions(annotation_dict, prediction_dict, plotter):
    plotter.plot_bounding_box_img([prediction_dict], annotation_dict)


def compare_ocr_numbers():
    pass


def compare_key_points():
    pass


def main(bbox_path, key_point_path, run_path, debug):

    annotation_dict = get_annotations_from_json(bbox_path, key_point_path)
    predictions_dict = get_predictions(run_path)

    assert set(predictions_dict.keys()) == set(annotation_dict.keys())

    for image_name in annotation_dict:
        # get corresponding image for plots
        image_path = predictions_dict[image_name][constants.ORIGINAL_IMG_KEY]
        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image)

        eval_path = os.path.join(run_path, image_name, "eval")
        os.makedirs(eval_path, exist_ok=True)
        plotter = EvalPlotter(eval_path, image)

        # compare gauge detection
        compare_gauge_detecions(
            annotation_dict[image_name][constants.GAUGE_DET_KEY],
            predictions_dict[image_name][constants.GAUGE_DET_KEY], plotter)

        # compare OCR number detection

        # compare OCR unit detection

        # compare key points

        # compare needle segmentations

        # maybe compare line fit and ellipse fit

    if debug:
        outfile_path = os.path.join(run_path, "true_annotations.json")
        write_json(outfile_path, annotation_dict)


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
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    main(args.bbox_true_path, args.keypoint_true_path, args.run_path,
         args.debug)
