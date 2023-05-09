import argparse
import os
import time
import sys
import json

import matplotlib
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

# pylint: disable=wrong-import-position
from key_point_dataset import KeypointImageDataSet, \
    IMG_PATH, LABEL_PATH, TRAIN_PATH, RUN_PATH, custom_transforms
from key_point_extraction import full_key_point_extraction, key_point_metrics,  \
        MEAN_DIST_KEY, PCK_KEY, NON_ASSIGNED_KEY
from model import load_model, N_HEATMAPS

matplotlib.use('Agg')

HEATMAP_PREFIX = "H_"
KEY_POINT_PREFIX = "K_"

HEATMAP_DIR = "heatmaps"
KEYPOINT_DIR = "key_points"

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
        key_point_metrics_dict = {}
        for index, data in enumerate(dataset):
            print(index)
            image, original_image, annotation = data

            image_name = dataset.get_name(index)

            heatmaps = self.model(image.unsqueeze(0))
            print("inference done")
            # take it as numpy array and decrease dimension by one
            heatmaps = heatmaps.detach().numpy().squeeze(0)

            # plot the heatmaps in the run folder
            heatmap_file_path = os.path.join(
                path, HEATMAP_DIR, HEATMAP_PREFIX + image_name + '.jpg')
            plot_heatmaps(heatmaps, annotation, heatmap_file_path)

            # Extract key points
            key_points_predicted = full_key_point_extraction(heatmaps,
                                                             threshold=0.6)
            key_points_true = full_key_point_extraction(
                annotation.detach().numpy(), threshold=0.95)

            print("key points extracted")

            key_point_metrics_dict[image_name] = key_point_metrics(
                key_points_predicted[1], key_points_true[1])

            # plot extracted key points
            key_point_file_path = os.path.join(
                path, KEYPOINT_DIR, KEY_POINT_PREFIX + image_name + '.jpg')
            #resize original image as well
            original_image_tensor = custom_transforms(train=False,
                                                      image=original_image)
            plot_key_points(original_image_tensor, key_points_predicted,
                            key_points_true, key_point_file_path)

        # Evaluate total metrics and save them to file
        total_mean_dist = 0
        total_pck = 0
        total_non_assigned = 0
        n_tests = len(key_point_metrics_dict)
        for single_metrics_dict in key_point_metrics_dict.values():
            total_mean_dist += single_metrics_dict[MEAN_DIST_KEY] / n_tests
            total_pck += single_metrics_dict[PCK_KEY] / n_tests
            total_non_assigned += single_metrics_dict[
                NON_ASSIGNED_KEY] / n_tests

        full_metrics_dict = {
            MEAN_DIST_KEY: total_mean_dist,
            PCK_KEY: total_pck,
            NON_ASSIGNED_KEY: total_non_assigned,
            "Individual results": key_point_metrics_dict
        }

        metrics_file_path = os.path.join(path, "key_point_metrics.json")
        full_metrics_json = json.dumps(full_metrics_dict, indent=4)
        with open(metrics_file_path, "w") as outfile:
            outfile.write(full_metrics_json)

    def validate(self):
        run_path = os.path.join(self.base_path, RUN_PATH + '_' + self.time_str)
        train_path = os.path.join(run_path, TRAIN_PATH)
        val_path = os.path.join(run_path, VAL_PATH)

        os.makedirs(run_path, exist_ok=True)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        os.makedirs(os.path.join(train_path, HEATMAP_DIR))
        os.makedirs(os.path.join(train_path, KEYPOINT_DIR))
        os.makedirs(os.path.join(val_path, HEATMAP_DIR))
        os.makedirs(os.path.join(val_path, KEYPOINT_DIR))

        self.validate_set(val_path, self.val_dataset)
        self.validate_set(train_path, self.train_dataset)


def plot_heatmaps(heatmaps1, heatmaps2, filename):
    plt.figure(figsize=(12, 8))

    if N_HEATMAPS == 1:
        plt.subplot(2, 1, 1)
        heatmap_plot = plt.imshow(heatmaps1.squeeze(0),
                                  cmap=plt.cm.viridis,
                                  vmin=0,
                                  vmax=1)
        plt.colorbar(heatmap_plot, shrink=0.5)
        plt.title('Predicted Heatmap')

        plt.subplot(2, 1, 2)
        heatmap_plot = plt.imshow(heatmaps2.squeeze(0),
                                  cmap=plt.cm.viridis,
                                  vmin=0,
                                  vmax=1)
        plt.colorbar(heatmap_plot, shrink=0.5)
        plt.title('True Heatmap')

    else:
        titles = ['Start', 'Middle', 'End']

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        plt.subplots_adjust(wspace=0.2,
                            hspace=0,
                            left=0.1,
                            right=0.9,
                            top=0.9,
                            bottom=0.1)
        for i in range(3):
            im = axs[0, i].imshow(heatmaps1[i],
                                  cmap=plt.cm.viridis,
                                  vmin=0,
                                  vmax=1)
            axs[0, i].set_title(f'Predicted Heatmap {titles[i]}')
        im1 = im
        for i in range(3):
            im = axs[1, i].imshow(heatmaps2[i],
                                  cmap=plt.cm.viridis,
                                  vmin=0,
                                  vmax=1)
            axs[1, i].set_title(f'True Heatmap {titles[i]}')
        im2 = im
        fig.colorbar(im1, ax=axs[0, :], shrink=0.8)
        fig.colorbar(im2, ax=axs[1, :], shrink=0.8)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_key_points(image, key_points_pred, key_points_true, filename):
    plt.figure(figsize=(12, 8))

    image = image.permute(1, 2, 0)

    if N_HEATMAPS == 1:
        key_points = key_points_pred[0]
        plt.subplot(2, 1, 1)
        plt.imshow(image)
        plt.scatter(key_points[:, 0],
                    key_points[:, 1],
                    s=50,
                    c='red',
                    marker='x')
        plt.title('Predicted Key Point')

        key_points = key_points_true[0]
        plt.subplot(2, 1, 2)
        plt.imshow(image)
        plt.scatter(key_points[:, 0],
                    key_points[:, 1],
                    s=50,
                    c='red',
                    marker='x')
        plt.title('True Key Point')
    else:
        titles = ['Start', 'Middle', 'End']
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
    plt.close()


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
