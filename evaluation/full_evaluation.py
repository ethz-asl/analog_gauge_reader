import argparse
import json
import os

import constants


def get_annotations_from_label_data(data):

    annotation_dict = {}

    for data_point in data:

        bbox_annotations = {}
        # Get image name. We have image name in format :
        # /data/upload/1/222ae49e-1_cropped_000001_jpg.rf.c7410b0b01b2bc3a6cdff656618a3015.jpg
        # get rid of everything before the '-'
        idx = data_point['data']['image'].find('-') + 1
        image_name = data_point['data']['image'][idx:]

        bbox_annotations[constants.OCR_NUM_KEY] = []
        bbox_annotations[constants.OCR_UNIT_KEY] = []
        bbox_annotations[constants.GAUGE_DET_KEY] = []

        for annotation in data_point['annotations'][0]['result']:

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


def get_annoations_from_json(annotation_path):
    with open(annotation_path, 'r') as file:
        bbox_true_dict = json.load(file)

    return get_annotations_from_label_data(bbox_true_dict)


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


def main(annotation_path, run_path, debug):

    annotation_dict = get_annoations_from_json(annotation_path)
    predictions_dict = get_predictions(run_path)

    print(predictions_dict)

    if debug:
        outfile_path = os.path.join(run_path, "true_bbox.json")
        write_json(outfile_path, annotation_dict)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_true_path',
                        type=str,
                        required=True,
                        help="Path to json file with labels for bbox")
    parser.add_argument('--run_path',
                        type=str,
                        required=True,
                        help="Path to run folder")
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    main(args.bbox_true_path, args.run_path, args.debug)
