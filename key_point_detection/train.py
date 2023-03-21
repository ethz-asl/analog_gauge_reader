import argparse
import os
import time

import numpy as np
import timm
import torch
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import Decoder

# Constants
N_HEATMAPS = 3
N_CHANNELS = 50  # Number of intermediate channels for Nonlinearity
INPUT_SIZE = (224, 224)

TRAIN_PATH = 'train'
VAL_PATH = 'val'
TEST_PATH = 'test'
IMG_PATH = 'images'
LABEL_PATH = 'labels'
RUN_PATH = 'runs'


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


class KeyPointTrain:
    def __init__(self, feature_extractor, base_path):

        image_folder = os.path.join(base_path, TRAIN_PATH, IMG_PATH)
        annotation_folder = os.path.join(base_path, TRAIN_PATH, LABEL_PATH)

        self.feature_extractor = feature_extractor

        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            FeatureExtractor(feature_extractor)
        ])
        self.train_dataset = KeypointImageDataSet(
            img_dir=image_folder,
            annotations_dir=annotation_folder,
            transform=self.transform)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=4)

        self.feature_shape = self._get_feature_shape()
        self.model = self._create_model()

    def _create_model(self):
        n_feature_channels = self.feature_shape[1]
        return Decoder(n_feature_channels, N_CHANNELS, INPUT_SIZE, N_HEATMAPS)

    def _get_feature_shape(self):
        input_shape = (1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
        x = torch.randn(input_shape)
        features = self.feature_extractor(x)
        return features.shape

    def get_train_dataset(self):
        return self.train_dataset

    def get_model(self):
        return self.model

    def train(self, num_epochs, learning_rate):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Train the model
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, annotations in self.train_dataloader:
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, annotations)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch {epoch + 1}: Loss = {running_loss / len(self.train_dataloader)}"
            )

        return self.model


class KeyPointVal:
    def __init__(self, feature_extractor, decoder_model, base_path):
        image_folder = os.path.join(base_path, VAL_PATH, IMG_PATH)
        annotation_folder = os.path.join(base_path, VAL_PATH, LABEL_PATH)

        self.base_path = base_path
        self.feature_extractor = feature_extractor
        self.decoder_model = decoder_model
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            FeatureExtractor(feature_extractor)
        ])

        self.val_dataset = KeypointImageDataSet(
            img_dir=image_folder,
            annotations_dir=annotation_folder,
            transform=self.transform)

    def validate(self, plot=False):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        run_path = os.path.join(self.base_path, RUN_PATH + '_' + time_str)
        os.mkdir(run_path)
        for index, data in enumerate(self.val_dataset):
            feature, annotation = data
            heatmaps = self.decoder_model(feature.unsqueeze(0))
            heatmaps = heatmaps.detach().numpy().squeeze(0)
            new_file_path = os.path.join(
                run_path,
                self.val_dataset.get_name(index) + '.jpg')

            plot_heatmaps(heatmaps, annotation, new_file_path, plot=plot)


def plot_heatmaps(heatmaps1, heatmaps2, filename=None, plot=False):
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

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    # Show the plots
    if plot:
        plt.show()


def main():
    args = read_args()

    # parameters for training
    encoder_model_name = args.encoder_model
    num_layers = args.layers  # number of layers of the feature extractor
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    base_path = args.data

    # fix seed for reproducibility
    torch.manual_seed(0)

    convnext_model = timm.create_model(encoder_model_name, pretrained=True)

    layer_list_encoder = [
        list(convnext_model.children())[0],
        *list(list(convnext_model.children())[1].children())[:num_layers]
    ]

    feature_extractor = nn.Sequential(*layer_list_encoder)

    # initialize trainer

    trainer = KeyPointTrain(feature_extractor, base_path)

    # train model
    decoder_model = trainer.train(num_epochs, learning_rate)

    validator = KeyPointVal(feature_extractor, decoder_model, base_path)
    validator.validate()


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_model',
                        type=str,
                        required=False,
                        default='convnext_base',
                        help="name of encoder model")
    parser.add_argument('--layers',
                        type=int,
                        required=False,
                        default=2,
                        help="Number of layers of feature extractor")
    parser.add_argument('--epochs',
                        type=int,
                        required=False,
                        default=50,
                        help="Number of epochs for training")
    parser.add_argument('--learning_rate',
                        type=float,
                        required=False,
                        default=3e-4,
                        help="Learning rate for training")
    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help="Base path of data")

    return parser.parse_args()


if __name__ == '__main__':
    main()
