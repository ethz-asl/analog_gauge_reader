import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from key_point_detection.key_point_extraction import full_key_point_extraction
from key_point_detection.model import load_model, INPUT_SIZE


class KeyPointInference:
    def __init__(self, model_path):

        self.model = load_model(model_path)

    def detect_key_points(self, image, debug):
        image = cv2.resize(image,
                           dsize=(INPUT_SIZE[1], INPUT_SIZE[0]),
                           interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image_t = torch.from_numpy(image)
        image_t = image_t.permute(2, 0, 1).unsqueeze(0)

        heatmaps = self.model(image_t)

        heatmaps = heatmaps.detach().numpy().squeeze(0)

        if debug:
            plot_heatmaps(heatmaps)

        key_point_list = full_key_point_extraction(heatmaps)

        return key_point_list


def plot_heatmaps(heatmaps):
    plt.figure(figsize=(12, 8))

    titles = ['Start', 'Middle', 'End']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(heatmaps[i], cmap=plt.cm.viridis)
        plt.title(f'Predicted Heatmap {titles[i]}')

    # Adjust the layout of the subplots
    plt.tight_layout()

    plt.show()
