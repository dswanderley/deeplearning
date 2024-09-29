import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from models import Generator
from train import generate_noise, LATENT_SIZE, HIDDEN_SIZE, IMAGE_SIZE

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(img: torch.Tensor) -> None:
    """
    Displays an image using Matplotlib.

    Args:
        img (torch.Tensor): The image tensor to display. Expected shape is [C, H, W].
    """
    img = img / 2 + 0.5  # De-normalize from [-1, 1] to [0, 1]
    npimg = img.cpu().numpy()  # Ensure the image is on CPU for visualization
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.show()


def inference(generator: Generator, latent_size: int, num_samples: int = 16) -> None:
    """
    Generates and visualizes images using the trained generator model.

    Args:
        generator (Generator): The trained generator model.
        latent_size (int): The size of the noise vector used to generate images.
        num_samples (int): The number of images to generate. Default is 16.
    """
    noise = generate_noise(num_samples, latent_size).to(device)  # Move noise to GPU if available
    fake_images: torch.Tensor = generator(noise)
    fake_images: torch.Tensor = fake_images.view(fake_images.size(0), 1, 28, 28)  # Reshape for visualization
    imshow(vutils.make_grid(fake_images.detach().cpu(), nrow=4))  # Ensure the images are on CPU for display


def load_model_and_run(generator: Generator, latent_size: int, model_path: str = 'gan/generator.pth', num_samples: int = 16) -> None:
    """
    Loads the trained generator model and runs inference.

    Args:
        generator (Generator): The generator model to load weights into.
        latent_size (int): The size of the noise vector used to generate images.
        model_path (str): The path to the saved model file. Default is 'gan/generator.pth'.
        num_samples (int): The number of images to generate. Default is 16.
    """
    generator.load_state_dict(torch.load(model_path, weights_only=True))
    generator.to(device)  # Move generator to GPU if available
    generator.eval()
    inference(generator, latent_size, num_samples)


if __name__ == "__main__":
    # Generator model
    generator = Generator(LATENT_SIZE, HIDDEN_SIZE, IMAGE_SIZE).to(device)  # Move generator to GPU if available
    # Load model from file and run
    load_model_and_run(generator, LATENT_SIZE)
