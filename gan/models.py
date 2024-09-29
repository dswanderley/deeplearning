import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    The Generator model for the GAN.

    Args:
        input_size (int): The size of the input noise vector.
        hidden_size (int): The size of the hidden layers.
        output_size (int): The size of the output (flattened image size).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Output should be in the range [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.

        Args:
            x (torch.Tensor): Input noise tensor.
        Returns:
            torch.Tensor: Generated image tensor.
        """
        return self.model(x)


class Discriminator(nn.Module):
    """
    The Discriminator model for the GAN.

    Args:
        input_size (int): The size of the input (flattened image size).
        hidden_size (int): The size of the hidden layers.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Probability tensor (real or fake).
        """
        return self.model(x)
