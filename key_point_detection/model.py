from torch import nn
import torch

ENCODER_MODEL_NAME = 'dinov2_vits14'

N_HEATMAPS = 3
N_CHANNELS = 50  # Number of intermediate channels for Nonlinearity
INPUT_SIZE = (224, 224)

DINO_CHANNELS = 384


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2',
                                    ENCODER_MODEL_NAME,
                                    pretrained=pretrained)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # pylint: disable=no-self-use
    def get_number_output_channels(self):
        return DINO_CHANNELS

    def forward(self, x):
        # pylint: disable=unused-variable
        B, C, H, W = x.shape
        with torch.no_grad():
            x = self.model.forward_features(x)['x_norm_patchtokens']
        width_out = W // 14
        height_out = H // 14
        return x.reshape(B, height_out, width_out,
                         DINO_CHANNELS).detach().permute(0, 3, 1, 2)


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
    n_feature_channels = encoder.get_number_output_channels()
    decoder = Decoder(n_feature_channels, N_CHANNELS, INPUT_SIZE, N_HEATMAPS)

    model = EncoderDecoder(encoder, decoder)
    model.load_state_dict(torch.load(model_path))
    return model
