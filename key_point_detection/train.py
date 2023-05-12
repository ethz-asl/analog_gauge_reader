import argparse
import os
import time
import sys
import logging
import json

import torch

from torch import nn, optim
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(parent_dir)

# pylint: disable=wrong-import-position
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

        self.train_dataset = KeypointImageDataSet(
            img_dir=image_folder,
            annotations_dir=annotation_folder,
            train=True,
            val=False)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=4)

        self.decoder = self._create_decoder()

        self.criterion = nn.BCELoss()

        self.full_model = EncoderDecoder(self.feature_extractor, self.decoder)

        self.loss = {}

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

        optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=50)

        # Train the model
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, annotations in self.train_dataloader:
                # Forward pass
                inputs, annotations = inputs.to(device), annotations.to(device)
                outputs = self.full_model(inputs)
                loss = self.criterion(outputs, annotations)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            loss = running_loss / len(self.train_dataloader)
            self.loss[epoch + 1] = loss

            # print new learning rate and loss
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(loss)
            after_lr = optimizer.param_groups[0]["lr"]
            loss_msg = f"Epoch {epoch + 1}: Loss = {loss}, lr {before_lr} -> {after_lr}"
            print(loss_msg)
            logging.info(loss_msg)

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

    # Setup run directory
    time_str = time.strftime("%Y%m%d-%H%M%S")
    run_path = os.path.join(base_path, RUN_PATH + '_' + time_str)
    os.makedirs(run_path, exist_ok=True)

    # Setup logger
    log_path = os.path.join(run_path, "run.log")

    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

    # initialize trainer
    if debug:
        print("initializing trainer")

    trainer = KeyPointTrain(base_path, debug)
    if debug:
        print("initialized trainer successfully")

    # train model
    logging.info("Start training")
    if debug:
        print("start training")

    trainer.train(num_epochs, learning_rate)

    logging.info("Finished training")

    model = trainer.get_full_model()

    # save model
    model_path = os.path.join(run_path, f"model_{time_str}.pt")
    torch.save(model.state_dict(), model_path)

    # save loss file
    loss_path = os.path.join(run_path, "loss.json")
    loss_json = json.dumps(trainer.loss, indent=4)
    with open(loss_path, "w") as outfile:
        outfile.write(loss_json)

    # save parameters to text file
    params = {
        'encoder': ENCODER_MODEL_NAME,
        'number of decoder channels': N_CHANNELS,
        'initial learning rate': learning_rate,
        'epochs': num_epochs,
        'batch size': BATCH_SIZE
    }

    param_file_path = os.path.join(run_path, "paramaters.txt")
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
