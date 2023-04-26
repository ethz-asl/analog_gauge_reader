import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt

from key_point_dataset import KeypointImageDataSet, \
    IMG_PATH, LABEL_PATH, TRAIN_PATH, RUN_PATH, custom_transforms
from key_point_extraction import full_key_point_extraction
from model import load_model

matplotlib.use('Agg')

HEATMAP_PREFIX = "H_"
KEY_POINT_PREFIX = "K_"

VAL_PATH = 'val'
TEST_PATH = 'test'


class KeyPointVal:
    def __init__(self, model, base_path, time_str=None):

        self.time_str = time_str if time_str is not None else time.strftime(
            "%Y%m%d-%H%M%S")

        train_image_folder = os.path.join(base_path, TRAIN_PATH, IMG_PATH)
        train_annotation_folder = os.path.join(base_path, TRAIN_PATH,
                                               LABEL_PATH)

        val_image_folder = os.path.join(base_path, VAL_PATH, IMG_PATH)
        val_annotation_folder = os.path.join(base_path, VAL_PATH, LABEL_PATH)

        self.base_path = base_path
        self.model = model

        self.train_dataset = KeypointImageDataSet(
            img_dir=train_image_folder,
            annotations_dir=train_annotation_folder,
            train=False,
            val=True)

        self.val_dataset = KeypointImageDataSet(
            img_dir=val_image_folder,
            annotations_dir=val_annotation_folder,
            train=False,
            val=True)

    def validate_set(self, path, dataset):
        for index, data in enumerate(dataset):
            print(index)
            image, original_image, annotation = data

            heatmaps = self.model(image.unsqueeze(0))
            print("inference done")
            # take it as numpy array and decrease dimension by one
            heatmaps = heatmaps.detach().numpy().squeeze(0)

            key_points = full_key_point_extraction(heatmaps, threshold=0.6)
            key_points_true = full_key_point_extraction(
                annotation.detach().numpy(), threshold=0.95)

            print("key points extracted")

            # plot the heatmaps in the run folder
            heatmap_file_path = os.path.join(
                path, HEATMAP_PREFIX + dataset.get_name(index) + '.jpg')
            plot_heatmaps(heatmaps, annotation, heatmap_file_path)
            key_point_file_path = os.path.join(
                path, KEY_POINT_PREFIX + dataset.get_name(index) + '.jpg')
            #resize original image as well
            original_image_tensor = custom_transforms(train=False,
                                                      image=original_image)
            plot_key_points(original_image_tensor, key_points, key_points_true,
                            key_point_file_path)

    def validate(self):
        run_path = os.path.join(self.base_path, RUN_PATH + '_' + self.time_str)
        train_path = os.path.join(run_path, TRAIN_PATH)
        val_path = os.path.join(run_path, VAL_PATH)

        os.makedirs(run_path, exist_ok=True)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        self.validate_set(train_path, self.train_dataset)
        self.validate_set(val_path, self.val_dataset)


def plot_heatmaps(heatmaps1, heatmaps2, filename):
    plt.figure(figsize=(12, 8))

    titles = ['Start', 'Middle', 'End']

    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.imshow(heatmaps1[i], cmap=plt.cm.viridis)
        plt.title(f'Predicted Heatmap {titles[i]}')

    for i in range(3):
        plt.subplot(2, 3, i + 4)
        plt.imshow(heatmaps2[i], cmap=plt.cm.viridis)
        plt.title(f'True Heatmap {titles[i]}')

    # Adjust the layout of the subplots
    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight')


def plot_key_points(image, key_points_pred, key_points_true, filename):
    plt.figure(figsize=(12, 8))

    titles = ['Start', 'Middle', 'End']

    image = image.permute(1, 2, 0)

    for i in range(3):
        key_points = key_points_pred[i]
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.scatter(key_points[:, 0],
                    key_points[:, 1],
                    s=50,
                    c='red',
                    marker='x')
        plt.title(f'Predicted Key Point {titles[i]}')

    for i in range(3):
        key_points = key_points_true[i]
        plt.subplot(2, 3, i + 4)
        plt.imshow(image)
        plt.scatter(key_points[:, 0],
                    key_points[:, 1],
                    s=50,
                    c='red',
                    marker='x')
        plt.title(f'True Key Point {titles[i]}')

    # Adjust the layout of the subplots
    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight')


def main():
    args = read_args()

    model_path = args.model_path
    base_path = args.data

    model = load_model(model_path)

    validator = KeyPointVal(model, base_path)
    validator.validate()


def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help="path to pytorch model")
    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help="Base path of data")

    return parser.parse_args()


if __name__ == '__main__':
    main()
