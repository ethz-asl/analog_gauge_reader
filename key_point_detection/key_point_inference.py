import torch

from key_point_extraction import full_key_point_extraction


class KeyPointInference:
    def __init__(self, model_path):
        self.model = torch.load(model_path)

    def detect_key_points(self, image):

        heatmaps = self.model(image)

        key_point_list = full_key_point_extraction(heatmaps)

        return key_point_list
