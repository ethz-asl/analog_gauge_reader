import os
import argparse


def get_files_from_folder(folder):
    filenames = {}
    for filename in os.listdir(folder):
        filenames[filename] = 0
    return filenames


def main():
    return


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        type=str,
                        required=True,
                        help="Path to folder with test images")
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    main()
