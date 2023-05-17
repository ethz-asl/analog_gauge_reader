import argparse
import json
import os

import constants


def get_annotations_bbox(data):

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

            if annotation['value']['rectanglelabels'][
                    0] == constants.OCR_NUM_KEY:
                bbox_annotations[constants.OCR_NUM_KEY].append({
                    k: annotation['value'][k]
                    for k in ('x', 'y', 'width', 'height')
                })

            if annotation['value']['rectanglelabels'][
                    0] == constants.OCR_UNIT_KEY:
                bbox_annotations[constants.OCR_UNIT_KEY].append({
                    k: annotation['value'][k]
                    for k in ('x', 'y', 'width', 'height')
                })

            if annotation['value']['rectanglelabels'][
                    0] == constants.GAUGE_DET_KEY:
                bbox_annotations[constants.GAUGE_DET_KEY].append({
                    k: annotation['value'][k]
                    for k in ('x', 'y', 'width', 'height')
                })

        annotation_dict[image_name] = bbox_annotations

    return annotation_dict


def get_annotations_keypoint(data):

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

            keypoint_annotations[constants.KEYPOINT_NOTCH_KEY].append(
                {k: annotation['value'][k]
                 for k in ('x', 'y')})

            if annotation['value']['keypointlabels'][
                    0] == constants.KEYPOINT_START_KEY:
                keypoint_annotations[constants.KEYPOINT_START_KEY].append(
                    {k: annotation['value'][k]
                     for k in ('x', 'y')})

            if annotation['value']['keypointlabels'][
                    0] == constants.KEYPOINT_END_KEY:
                keypoint_annotations[constants.KEYPOINT_END_KEY].append(
                    {k: annotation['value'][k]
                     for k in ('x', 'y')})

        annotation_dict[image_name] = keypoint_annotations

    return annotation_dict


def get_annotations_from_json(bbox_path, key_point_path):
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


def main(bbox_path, key_point_path, run_path, debug):

    annotation_dict = get_annotations_from_json(bbox_path, key_point_path)
    predictions_dict = get_predictions(run_path)

    print(predictions_dict)

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
