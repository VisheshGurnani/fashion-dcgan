import torch
from torch import nn


LATENT_DIM = 100


class Generator(nn.Module):
    """Project a 100D noise vector into a 28x28 grayscale image."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7, bias=False),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    """Classify a 28x28 grayscale image as real or generated."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images).view(-1)
