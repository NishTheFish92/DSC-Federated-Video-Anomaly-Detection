"""Small convolutional autoencoder: 1x64x64 frame -> 128x8x8 code -> 1x64x64 frame."""
from torch import nn


def down(c_in, c_out):
    # Stride-2 conv halves height and width.
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1), nn.ReLU())


def up(c_in, c_out):
    # Stride-2 transposed conv doubles height and width.
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), nn.ReLU())


class AutoEncoder(nn.Module):
    # No BatchNorm on purpose: every weight is a plain parameter, so FedAvg is a simple mean.
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(down(1, 32), down(32, 64), down(64, 128))
        self.decoder = nn.Sequential(
            up(128, 64), up(64, 32),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),  # outputs in [0, 1], like the input frames
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
