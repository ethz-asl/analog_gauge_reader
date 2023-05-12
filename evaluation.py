import os
import argparse
import json

from common import ERROR_FILE_NAME, RESULT_FILE_NAME, READING_KEY, FAILED, \
    OCR_NONE_DETECTED_KEY, OCR_ONLY_ONE_DETECTED_KEY, RANGE_KEY

PRED = 'prediction'
TRUTH = 'true_reading'
ABS_ERROR = 'total absolute error'
REL_ERROR = 'total relative error'
N_FAILED = 'number of failed predictions'
N_FAILED_OCR = 'number of failed OCR, less than 2 numbers detected'
COMPARSION = 'comparison'

ERRORS = 'errors'


def get_files_from_folder(folder):
    filenames = {}
    for filename in os.listdir(folder):
        filenames[filename] = 0
    return filenames


def get_errors(run_path):
    errors = {}
    for subdir in os.listdir(run_path):

        subdirectory = os.path.join(run_path, subdir)
        if os.path.isdir(subdirectory):

            result_file = os.path.join(subdirectory, ERROR_FILE_NAME)
            if os.path.isfile(result_file):
                with open(result_file, 'r') as file:
                    errors_dict = json.load(file)
                    errors[subdir] = errors_dict

            else:
                errors[subdir] = FAILED
                print("Error: No Error file found! \
                        Pipeline failed unexpectadly before error file could be made"
                      )

    outfile_path = os.path.join(run_path, "predictions_mean_errors.json")
    errors_json = json.dumps(errors, indent=4)
    with open(outfile_path, "w") as outfile:
        outfile.write(errors_json)

    return errors


def get_predictions(run_path):
    predictions = {}

    for subdir in os.listdir(run_path):

        subdirectory = os.path.join(run_path, subdir)
        if os.path.isdir(subdirectory):

            result_file = os.path.join(subdirectory, RESULT_FILE_NAME)
            if os.path.isfile(result_file):
                with open(result_file, 'r') as file:
                    result_dict = json.load(file)
                    predictions[subdir] = result_dict[0][READING_KEY]

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
    errors = get_errors(run_path)
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

    results[N_FAILED] = f"{n_failed} / {n_total}"

    n_ocr_failed = 0
    for individual_error_dict in errors.values():
        if OCR_NONE_DETECTED_KEY in individual_error_dict or \
            OCR_ONLY_ONE_DETECTED_KEY in individual_error_dict:
            n_ocr_failed += 1

    results[N_FAILED_OCR] = n_ocr_failed

    full_comparison = {}

    total_absolute_error = 0
    total_relative_error = 0
    for key in predictions:
        full_comparison[key] = {}
        full_comparison[key][PRED] = predictions[key]
        full_comparison[key][TRUTH] = true_readings[key][READING_KEY]
        unit_range = true_readings[key][RANGE_KEY]

        if predictions[key] != FAILED:
            absolute_error = abs(predictions[key] -
                                 true_readings[key][READING_KEY])
            relative_error = absolute_error / unit_range
            full_comparison[key][ABS_ERROR] = absolute_error
            full_comparison[key][REL_ERROR] = relative_error
            total_absolute_error += absolute_error / n_predicted
            total_relative_error += relative_error / n_predicted

    results[ABS_ERROR] = total_absolute_error
    results[REL_ERROR] = total_relative_error
    results[COMPARSION] = full_comparison

    results[ERRORS] = errors

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
