from torch import nn


class Decoder(nn.Module):
    def __init__(self, n_input_channels, n_inter_channels, out_size,
                 n_heatmaps):
        super().__init__()
        self.upsampling = nn.Sequential(
            nn.Upsample(size=out_size, mode='bilinear'))
        self.heatmaphead = nn.Sequential(
            nn.Conv2d(n_input_channels, n_inter_channels, (1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(n_inter_channels, n_heatmaps, (1, 1), bias=True),
            nn.Sigmoid())

    def forward(self, x):
        upsample = self.upsampling(x)
        heatmap = self.heatmaphead(upsample)
        return heatmap
