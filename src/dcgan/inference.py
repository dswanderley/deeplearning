import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from models import Generator


# Configurations
NGPU: int = 1
NZ: int = 100  # Size of the latent z vector (noise)
NGF: int = 64  # Size of feature maps in generator
NC: int = 3  # Number of image channels (3 for RGB)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def imshow(img: torch.Tensor) -> None:
    """
    Visualizes an image tensor using Matplotlib.

    Args:
        img (torch.Tensor): A tensor representing the image to be displayed.
                            It is expected to have values normalized between [-1, 1].
    """
    img = img / 2 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    npimg = img.cpu().numpy()  # Ensure the image is on CPU for visualization
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Transpose to match image format (H, W, C)
    plt.show()


def inference(generator: Generator, nz: int, model_path: str = './models/dcgan/generator_dcgan.pth', num_samples: int = 64) -> None:
    """
    Generates and visualizes images using a pre-trained generator model.

    Args:
        generator (Generator): The pre-trained generator model to use for inference.
        nz (int): Size of the latent noise vector used for generating images.
        model_path (str): Path to the saved generator model file. Defaults to './models/dcgan/generator_dcgan.pth'.
        num_samples (int): Number of samples (images) to generate. Defaults to 64.
    """
    # Load the trained generator weights
    generator.load_state_dict(torch.load(model_path))

    # Set the generator to evaluation mode
    generator.eval()

    # Generate random noise
    noise = torch.randn(num_samples, nz, 1, 1, device=device)

    # Generate fake images using the generator
    fake_images = generator(noise)

    # Visualize the generated images
    imshow(vutils.make_grid(fake_images.detach().cpu(), nrow=8, padding=2))


if __name__ == "__main__":
    # Initialize the generator model
    generator = Generator(ngpu=NGPU, nz=NZ, ngf=NGF, nc=NC).to(device)

    # Perform inference and visualize the generated images
    inference(generator, NZ)
