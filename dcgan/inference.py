import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from models import Generator


# Configurations
NGPU = 1
NZ = 100  # Size of the latent z vector (noise)
NGF = 64  # Size of feature maps in generator
NC = 3  # Number of image channels (3 for RGB)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize generator model
generator = Generator(ngpu=NGPU, nz=NZ, ngf=NGF, nc=NC).to(device)
generator.load_state_dict(torch.load('generator_dcgan.pth'))
generator.eval()


# Function to visualize generated images
def imshow(img: torch.Tensor) -> None:
    img = img / 2 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Generate and visualize images after training
def inference(generator: Generator, nz: int, num_samples: int = 64):
    noise = torch.randn(num_samples, nz, 1, 1, device=device)
    fake_images = generator(noise)
    imshow(vutils.make_grid(fake_images.detach().cpu(), nrow=8, padding=2))


if __name__ == "__main__":
    inference(generator, NZ)
