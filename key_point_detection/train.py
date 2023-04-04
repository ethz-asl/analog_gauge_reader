import argparse
import os
import time

import timm
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from key_point_dataset import FeatureExtractor, KeypointImageDataSet, \
    TRAIN_PATH, IMG_PATH, LABEL_PATH, INPUT_SIZE, N_HEATMAPS, N_CHANNELS
from key_point_validator import KeyPointVal
from model import Decoder


class KeyPointTrain:
    def __init__(self, feature_extractor, base_path, debug):

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
        self.decoder = self._create_decoder()

        self.debug = debug

    def _create_decoder(self):
        n_feature_channels = self.feature_shape[1]
        return Decoder(n_feature_channels, N_CHANNELS, INPUT_SIZE, N_HEATMAPS)

    def _get_feature_shape(self):
        input_shape = (1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
        x = torch.randn(input_shape)
        features = self.feature_extractor(x)
        return features.shape

    def train(self, num_epochs, learning_rate):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # Train the model
        for epoch in range(num_epochs):
            running_loss = 0.0
            for features, annotations in self.train_dataloader:
                # Forward pass
                outputs = self.decoder(features)
                loss = criterion(outputs, annotations)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}: Loss = {running_loss / len(self.train_dataloader)}, "
                f"lr {before_lr} -> {after_lr} ")

    def get_full_model(self):
        return nn.Sequential(self.feature_extractor, self.decoder)


def main():
    args = read_args()

    # parameters for training
    encoder_model_name = args.encoder_model
    num_layers = args.layers  # number of layers of the feature extractor
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    base_path = args.data
    val = args.val
    debug = args.debug
    # fix seed for reproducibility
    torch.manual_seed(0)

    if debug:
        print("load encoder model")

    encoder_model = timm.create_model(encoder_model_name, pretrained=True)

    if debug:
        print("encoder model loaded successfully")

    # only take first num_layers of encoder, determines depth of encoder
    layer_list_encoder = [
        list(encoder_model.children())[0],
        *list(list(encoder_model.children())[1].children())[:num_layers]
    ]

    feature_extractor = nn.Sequential(*layer_list_encoder)

    # initialize trainer
    if debug:
        print("initializing trainer")
    trainer = KeyPointTrain(feature_extractor, base_path, debug)
    if debug:
        print("initialized trainer successfully")

    # train model
    if debug:
        print("start training")
    trainer.train(num_epochs, learning_rate)
    model = trainer.get_full_model()

    # save model
    time_str = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(base_path, f"model_{time_str}.pt")
    torch.save(model, model_path)
    if val:
        validator = KeyPointVal(model, base_path)
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
    parser.add_argument(
        '--val',
        type=bool,
        required=False,
        default=False,
        help="Should validation be done after training, saving images")
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    main()
