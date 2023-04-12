from torch import nn
import torch
import timm

ENCODER_MODEL_NAME = 'convnext_base'
NUM_LAYERS = 2

N_HEATMAPS = 3
N_CHANNELS = 50  # Number of intermediate channels for Nonlinearity
INPUT_SIZE = (224, 224)


class Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        encoder_model = timm.create_model(ENCODER_MODEL_NAME,
                                          pretrained=pretrained)

        # only take first num_layers of encoder, determines depth of encoder
        layer_list_encoder = [
            list(encoder_model.children())[0],
            *list(list(encoder_model.children())[1].children())[:NUM_LAYERS]
        ]

        self.feature_extractor = nn.Sequential(*layer_list_encoder)

    def get_feature_shape(self):
        input_shape = (1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
        x = torch.randn(input_shape)
        features = self.feature_extractor(x)
        return features.shape

    def forward(self, x):
        return self.feature_extractor(x)


class Decoder(nn.Module):
    def __init__(self, n_input_channels, n_inter_channels, out_size,
                 n_heatmaps):
        super().__init__()
        self.upsampling = nn.Sequential(
            nn.Upsample(size=out_size, mode='bilinear', align_corners=False))
        self.heatmaphead = nn.Sequential(
            nn.Conv2d(n_input_channels, n_inter_channels, (1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(n_inter_channels, n_heatmaps, (1, 1), bias=True),
            nn.Sigmoid())

    def forward(self, x):
        upsample = self.upsampling(x)
        heatmap = self.heatmaphead(upsample)
        return heatmap


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_model(model_path):
    encoder = Encoder(pretrained=False)
    n_feature_channels = encoder.get_feature_shape()[1]
    decoder = Decoder(n_feature_channels, N_CHANNELS, INPUT_SIZE, N_HEATMAPS)

    model = EncoderDecoder(encoder, decoder)
    model.load_state_dict(torch.load(model_path))
    return model
