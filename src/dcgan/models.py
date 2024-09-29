import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Deep Convolutional Generator for generating images.

    Args:
        ngpu (int): Number of GPUs available. Can be set to 1 for using a single GPU.
        nz (int): Size of the input noise vector.
        ngf (int): Size of feature maps in the generator.
        nc (int): Number of channels in the generated image (3 for RGB).
    """
    def __init__(self, ngpu: int, nz: int, ngf: int, nc: int):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Output between [-1, 1]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class Discriminator(nn.Module):
    """
    Deep Convolutional Discriminator for classifying real vs. fake images.

    Args:
        ngpu (int): Number of GPUs available. Can be set to 1 for using a single GPU.
        nc (int): Number of channels in the input image (3 for RGB).
        ndf (int): Size of feature maps in the discriminator.
    """
    def __init__(self, ngpu: int, nc: int, ndf: int):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output between [0, 1]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)
