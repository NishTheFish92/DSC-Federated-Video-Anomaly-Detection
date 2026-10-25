"""Small convolutional autoencoder: Cx64x64 input -> 128x8x8 code -> Cx64x64 output.

C = 1 for a single frame, or C = STACK when consecutive frames are stacked as channels.
"""
from torch import nn


def down(c_in, c_out):
    # Stride-2 conv halves height and width.
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1), nn.ReLU())


def up(c_in, c_out):
    # Stride-2 transposed conv doubles height and width.
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), nn.ReLU())


class AutoEncoder(nn.Module):
    # No BatchNorm on purpose: every weight is a plain parameter, so FedAvg is a simple mean.
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels  # frames per input; evaluate.py reads this to stack test frames
        self.encoder = nn.Sequential(down(channels, 32), down(32, 64), down(64, 128))
        self.decoder = nn.Sequential(
            up(128, 64), up(64, 32),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # outputs in [0, 1], like the input frames
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
