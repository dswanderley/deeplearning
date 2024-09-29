import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from models import Generator
from train import generate_noise, LATENT_SIZE, HIDDEN_SIZE, IMAGE_SIZE

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to visualize generated images
def imshow(img):
    img = img / 2 + 0.5  # De-normalize from [-1, 1] to [0, 1]
    npimg = img.cpu().numpy()  # Ensure the image is on CPU for visualization
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.show()


# Generate and visualize images after training
def inference(generator, latent_size, num_samples=16):
    noise = generate_noise(num_samples, latent_size).to(device)  # Move noise to GPU if available
    fake_images = generator(noise)
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)  # Reshape for visualization
    imshow(vutils.make_grid(fake_images.detach().cpu(), nrow=4))  # Ensure the images are on CPU for display


# Load model and run inference
def load_model_and_run(generator, latent_size, model_path='gan/generator.pth', num_samples=16):
    # Load trained weights
    generator.load_state_dict(torch.load(model_path, weights_only=True))
    generator.to(device)  # Move generator to GPU if available
    generator.eval()
    inference(generator, latent_size, num_samples)


if __name__ == "__main__":
    # Generator model
    generator = Generator(LATENT_SIZE, HIDDEN_SIZE, IMAGE_SIZE).to(device)  # Move generator to GPU if available
    # Load model from file and run
    load_model_and_run(generator, LATENT_SIZE)
