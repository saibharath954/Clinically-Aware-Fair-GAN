import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    CAF-GAN Discriminator Network (Critic for WGAN-GP).
    Takes a 256x256 grayscale image and outputs a single scalar score.
    Uses InstanceNorm instead of BatchNorm for stability with WGAN-GP.
    No final activation function.
    """
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Input: channels x 256 x 256
            nn.Conv2d(channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 32 x 128 x 128
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 64 x 64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 32 x 32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 16 x 16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 8 x 8
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 1024 x 4 x 4
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False)
            # Output: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)