import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from train import generate_noise, generator, LATENT_SIZE


# Function to visualize generated images
def imshow(img):
    img = img / 2 + 0.5  # De-normalize from [-1, 1] to [0, 1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.show()


# Generate and visualize images after training
def evaluate(generator, latent_size, num_samples=16):
    noise = generate_noise(num_samples, latent_size)
    fake_images = generator(noise)
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)  # Reshape for visualization
    imshow(vutils.make_grid(fake_images.detach(), nrow=4))


def load_model_and_evaluate(generator, latent_size, model_path='generator.pth', num_samples=16):
    # Load trained weights
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    evaluate(generator, latent_size, num_samples)


# Call evaluation after training
load_model_and_evaluate(generator, LATENT_SIZE)
