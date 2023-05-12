import os
import json
import argparse

from common import RANGE_KEY, READING_KEY, MEASURE_UNIT_KEY


def get_files_from_folder(folder):
    filenames = {}
    for filename in sorted(os.listdir(folder)):
        filenames[filename] = {
            READING_KEY: 0,
            RANGE_KEY: 0,
            MEASURE_UNIT_KEY: "bar"
        }
    return filenames


def main(path):
    folder = os.path.join(path, "images")
    image_names = get_files_from_folder(folder)

    names_json = json.dumps(image_names, indent=4)
    outfile_path = os.path.join(path, "true_readings.json")
    with open(outfile_path, "w") as outfile:
        outfile.write(names_json)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        type=str,
                        required=True,
                        help="Path to folder with test images")
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    main(args.path)
