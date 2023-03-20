import argparse
import os
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
    def __init__(self, feature_extractor, image_folder, annotation_folder):
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


def compare_heatmap(trainer, features, annotation, filename, plot=False):
    heatmaps = trainer.get_model()(features.unsqueeze(0))
    heatmaps = heatmaps.detach().numpy().squeeze()

    true_test_label = annotation

    plot_heatmaps(heatmaps, true_test_label, filename, plot)


def main():

    args = read_args()

    # parameters for training
    encoder_model_name = args.encoder_model
    num_layers = args.layers  # number of layers of the feature extractor
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    train_image_path = args.train_img_path
    train_label_path = args.train_label_path

    image_out = args.img_out

    # for debugging to see result of single test image
    test_image_path = args.test_img_path
    test_label_path = args.test_label_path

    if test_image_path is not None:
        assert test_label_path is not None

    # fix seed for reproducibility
    torch.manual_seed(0)

    convnext_model = timm.create_model(encoder_model_name, pretrained=True)

    layer_list_encoder = [
        list(convnext_model.children())[0],
        *list(list(convnext_model.children())[1].children())[:num_layers]
    ]

    feature_extractor = nn.Sequential(*layer_list_encoder)

    # initialize trainer
    trainer = KeyPointTrain(feature_extractor, train_image_path,
                            train_label_path)

    # train model
    decoder_model = trainer.train(num_epochs, learning_rate)

    # try out trained model on test image, plot result and compare it to true label
    test_img = Image.open(test_image_path).convert("RGB")
    features = trainer.transform(test_img)
    heatmaps = decoder_model(features.unsqueeze(0))
    heatmaps = heatmaps.detach().numpy().squeeze(0)

    true_test_label = np.load(test_label_path)
    plot_heatmaps(heatmaps, true_test_label, plot=True)

    # plot and save results on training data to see if results are satisfactory on training images.
    if image_out is not None:
        dataset = trainer.get_train_dataset()
        for index, data in enumerate(dataset):
            feature, annotation = data
            new_file_path = image_out + dataset.get_name(index) + '.jpg'
            compare_heatmap(trainer, feature, annotation, new_file_path)


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
                        help="Path to input image")

    parser.add_argument('--train_img_path',
                        type=str,
                        required=True,
                        help="Path to train images")
    parser.add_argument('--train_label_path',
                        type=str,
                        required=True,
                        help="Path to train labels")
    parser.add_argument('--test_img_path',
                        type=str,
                        required=False,
                        help="Path to test images")
    parser.add_argument('--test_label_path',
                        type=str,
                        required=False,
                        help="Path to test label")
    parser.add_argument('--img_out',
                        type=str,
                        required=False,
                        help="Path to save predictions of training images to")
    return parser.parse_args()


if __name__ == '__main__':
    main()
