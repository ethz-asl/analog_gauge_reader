import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

# Constants
N_HEATMAPS = 3
N_CHANNELS = 50  # Number of intermediate channels for Nonlinearity
INPUT_SIZE = (224, 224)

TRAIN_PATH = 'train'
IMG_PATH = 'images'
LABEL_PATH = 'labels'
RUN_PATH = 'runs'
VAL_PATH = 'val'
TEST_PATH = 'test'

HEATMAP_PREFIX = "H_"
KEY_POINT_PREFIX = "K_"


class KeypointImageDataSet(Dataset):
    def __init__(self,
                 img_dir,
                 annotations_dir,
                 transform=None,
                 target_transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.target_transform = target_transform

        self.image_files = sorted(os.listdir(img_dir))
        self.annotation_files = sorted(os.listdir(annotations_dir))

        assert len(self.image_files) == len(self.annotation_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.image_files[index])
        annotation_path = os.path.join(self.annotations_dir,
                                       self.annotation_files[index])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        annotations = np.load(annotation_path)

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        # Convert to tensors
        annotations = torch.tensor(annotations, dtype=torch.float32)

        return image, annotations

    def get_name(self, index):
        return self.image_files[index][:-4]


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, img):
        with torch.no_grad():
            features = self.model(img.unsqueeze(0))
        return features.squeeze(0)
