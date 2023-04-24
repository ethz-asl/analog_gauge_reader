import argparse
import os
import time

import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from key_point_dataset import RUN_PATH, KeypointImageDataSet, \
    TRAIN_PATH, IMG_PATH, LABEL_PATH
from key_point_validator import KeyPointVal
from model import ENCODER_MODEL_NAME, Encoder, Decoder, EncoderDecoder, \
    INPUT_SIZE, N_HEATMAPS, N_CHANNELS

BATCH_SIZE = 8


class KeyPointTrain:
    def __init__(self, base_path, debug):

        self.debug = debug

        image_folder = os.path.join(base_path, TRAIN_PATH, IMG_PATH)
        annotation_folder = os.path.join(base_path, TRAIN_PATH, LABEL_PATH)

        self.feature_extractor = Encoder(pretrained=True)

        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.train_dataset = KeypointImageDataSet(
            img_dir=image_folder,
            annotations_dir=annotation_folder,
            transform=self.transform)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=4)

        self.decoder = self._create_decoder()

        self.full_model = EncoderDecoder(self.feature_extractor, self.decoder)

    def _create_decoder(self):
        n_feature_channels = self.feature_extractor.get_number_output_channels(
        )
        if self.debug:
            print(f"Number of feature channels is {n_feature_channels}")
        return Decoder(n_feature_channels, N_CHANNELS, INPUT_SIZE, N_HEATMAPS)

    def train(self, num_epochs, learning_rate):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.debug:
            print(f"Using {device} device")

        self.full_model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=10)

        # Train the model
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, annotations in self.train_dataloader:
                # Forward pass
                inputs, annotations = inputs.to(device), annotations.to(device)
                outputs = self.full_model(inputs)
                loss = criterion(outputs, annotations)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            loss = running_loss / len(self.train_dataloader)

            # print new learning rate and loss
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(loss)
            after_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}: Loss = {loss}, "
                  f"lr {before_lr} -> {after_lr} ")

        print('Finished Training')

    def get_full_model(self):
        return self.full_model


def main():
    args = read_args()

    # parameters for training
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    base_path = args.data
    val = args.val
    debug = args.debug
    # fix seed for reproducibility
    torch.manual_seed(0)

    # initialize trainer
    if debug:
        print("initializing trainer")

    trainer = KeyPointTrain(base_path, debug)
    if debug:
        print("initialized trainer successfully")

    # train model
    if debug:
        print("start training")
    trainer.train(num_epochs, learning_rate)
    model = trainer.get_full_model()

    time_str = time.strftime("%Y%m%d-%H%M%S")
    run_path = RUN_PATH + '_' + time_str
    os.makedirs(os.path.join(base_path, run_path), exist_ok=True)

    # save model
    model_path = os.path.join(base_path, run_path, f"model_{time_str}.pt")
    torch.save(model.state_dict(), model_path)

    # save parameters to text file
    params = {
        'encoder': ENCODER_MODEL_NAME,
        'number of decoder channels': N_CHANNELS,
        'initial learning rate': learning_rate,
        'epochs': num_epochs,
        'batch size': BATCH_SIZE
    }

    param_file_path = os.path.join(base_path, run_path, "paramaters.txt")
    write_parameter_file(param_file_path, params)

    if val:
        validator = KeyPointVal(model, base_path, time_str)
        validator.validate()


def write_parameter_file(filename, params):
    with open(filename, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def read_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    main()
