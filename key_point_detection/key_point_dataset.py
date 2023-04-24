import os
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from model import INPUT_SIZE

# Constants

TRAIN_PATH = 'train'
IMG_PATH = 'images'
LABEL_PATH = 'labels'
RUN_PATH = 'runs'
VAL_PATH = 'val'
TEST_PATH = 'test'

HEATMAP_PREFIX = "H_"
KEY_POINT_PREFIX = "K_"


class KeypointImageDataSet(Dataset):
    def __init__(self, img_dir, annotations_dir, train=False, val=False):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir

        self.image_files = sorted(os.listdir(img_dir))
        self.annotation_files = sorted(os.listdir(annotations_dir))

        self.train = train
        self.val = val

        assert len(self.image_files) == len(self.annotation_files)

    def my_segmentation_transforms(self, image, annotation):
        random.seed(0)

        resize = transforms.Resize(INPUT_SIZE,
                                   transforms.InterpolationMode.BILINEAR)
        image = resize(image)
        annotation = resize(annotation)

        toTensor = transforms.ToTensor()
        # random crop image and annotation
        if self.train:
            if random.random() > 0.5:
                new_size = int(1.2 * INPUT_SIZE[0])
                resize = transforms.Resize(
                    new_size, transforms.InterpolationMode.BILINEAR)
                # increase size
                image = resize(image)
                annotation = resize(annotation)

                top = random.randint(0, new_size - INPUT_SIZE[0])
                left = random.randint(0, new_size - INPUT_SIZE[0])
                image = TF.crop(image, top, left, INPUT_SIZE[0], INPUT_SIZE[1])
                annotation = TF.crop(annotation, top, left, INPUT_SIZE[0],
                                     INPUT_SIZE[1])

        return toTensor(image), toTensor(annotation)

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
        annotations_image = annotations_np_to_img(annotations)

        transformed_image, transformed_annotation = self.my_segmentation_transforms(
            image, annotations_image)

        # Convert to tensors
        if self.val:
            return transformed_image, image, transformed_annotation
        return transformed_image, transformed_annotation

    def get_name(self, index):
        return self.image_files[index][:-4]


def annotations_np_to_img(annotations):
    annotations = annotations.transpose(1, 2, 0)
    annotations = (annotations * 255).astype(np.uint8)
    return Image.fromarray(annotations, mode='RGB')
