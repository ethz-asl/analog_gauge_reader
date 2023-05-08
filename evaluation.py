import os
import argparse
import json

from common import RESULT_FILE_NAME, PREDICTION_KEY

FAILED = 'Failed'
PRED = 'prediction'
TRUTH = 'true_reading'
ABS_ERROR = 'total absolute error'
REL_ERROR = 'total relative error'
N_FAILED = 'number of failed predictions'
COMPARSION = 'comparison'


def get_files_from_folder(folder):
    filenames = {}
    for filename in os.listdir(folder):
        filenames[filename] = 0
    return filenames


def get_predictions(run_path):
    predictions = {}

    for subdir in os.listdir(run_path):

        subdirectory = os.path.join(run_path, subdir)
        if os.path.isdir(subdirectory):

            result_file = os.path.join(subdirectory, RESULT_FILE_NAME)
            if os.path.isfile(result_file):
                with open(result_file, 'r') as file:
                    result_dict = json.load(file)
                    predictions[subdir] = result_dict[0][PREDICTION_KEY]

            else:
                predictions[subdir] = FAILED
                print("Error: No Prediction file found! \
                        Pipeline failed before prediction could be made")

    outfile_path = os.path.join(run_path, "predictions_readings.json")
    predictions_json = json.dumps(predictions, indent=4)
    with open(outfile_path, "w") as outfile:
        outfile.write(predictions_json)

    return predictions


def main(run_path, true_readings_path):
    predictions = get_predictions(run_path)
    with open(true_readings_path, 'r') as file:
        true_readings = json.load(file)

    assert set(predictions.keys()) == set(true_readings.keys())

    results = {}

    n_failed = 0
    for key in predictions:
        if predictions[key] == FAILED:
            n_failed += 1
    n_total = len(predictions)
    n_predicted = n_total - n_failed

    results[N_FAILED] = f"{n_failed} / {n_predicted}"

    full_comparison = {}

    total_absolute_error = 0
    for key in predictions:
        full_comparison[key] = {}
        full_comparison[key][PRED] = predictions[key]
        full_comparison[key][TRUTH] = true_readings[key]

        if predictions[key] != FAILED:
            absolute_error = abs(predictions[key] - true_readings[key])
            full_comparison[key][ABS_ERROR] = absolute_error
            total_absolute_error += absolute_error / n_predicted

    results[ABS_ERROR] = total_absolute_error
    results[COMPARSION] = full_comparison

    outfile_path = os.path.join(run_path, "evaluation.json")
    results_json = json.dumps(results, indent=4)
    with open(outfile_path, "w") as outfile:
        outfile.write(results_json)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path',
                        type=str,
                        required=True,
                        help="Path to pipeline run on test images")
    parser.add_argument('--true_readings_path',
                        type=str,
                        required=True,
                        help="Path to json file with true readings")
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    main(args.run_path, args.true_readings_path)
