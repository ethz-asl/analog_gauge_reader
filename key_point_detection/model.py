from torch import nn


class Decoder(nn.Module):
    def __init__(self, n_channels, output_w, output_h, n_heatmaps):
        super().__init__()
        self.upsampling = nn.Sequential(
            nn.Upsample(size=(output_w, output_h), mode='bilinear'))
        self.heatmaphead = nn.Sequential(
            nn.Conv2d(n_channels, n_heatmaps, (1, 1), bias=True), nn.Sigmoid())

    def forward(self, x):
        upsample = self.upsampling(x)
        heatmap = self.heatmaphead(upsample)
        return heatmap
