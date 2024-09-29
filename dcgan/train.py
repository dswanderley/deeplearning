import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator

# Configurations
BATCH_SIZE = 128
LATENT_SIZE = 100  # Size of the latent z vector (noise)
IMAGE_SIZE = 64  # Image size for CIFAR-10 (64x64)
NC = 3  # Number of channels in the image (3 for RGB)
NGF = 64  # Size of feature maps in the generator
NDF = 64  # Size of feature maps in the discriminator
NUM_EPOCHS = 25
LEARNING_RATE = 0.0002
BETA1 = 0.5  # Beta1 for Adam optimizer
NGPU = 1  # Number of GPUs available


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_noise(batch_size: int, latent_size: int) -> torch.Tensor:
    """
    Generates a random noise tensor for the generator.

    Args:
        batch_size (int): Number of noise vectors to generate.
        latent_size (int): Size of each noise vector (z dimension).

    Returns:
        torch.Tensor: Random noise tensor of shape (batch_size, latent_size, 1, 1).
    """
    return torch.randn(batch_size, latent_size, 1, 1, device=device)


def train(generator: Generator, discriminator: Discriminator, train_loader: DataLoader,
          optimizer_g: torch.optim.Optimizer, optimizer_d: torch.optim.Optimizer,
          criterion: nn.BCELoss, num_epochs: int, latent_size: int) -> None:
    """
    Trains the DCGAN using the generator and discriminator models.

    Args:
        generator (Generator): The generator model.
        discriminator (Discriminator): The discriminator model.
        train_loader (DataLoader): DataLoader for the CIFAR-10 training dataset.
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
        criterion (nn.BCELoss): Loss function (binary cross-entropy).
        num_epochs (int): Number of epochs to train.
        latent_size (int): Size of the latent vector (noise) for the generator.
    """

    best_g_loss = float('inf')  # Initialize the best generator loss with infinity
    best_generator_path = 'best_generator_dcgan.pth'

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Create real and fake labels
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # Train discriminator on real images
            optimizer_d.zero_grad()
            output_real = discriminator(real_images).view(-1)
            d_loss_real = criterion(output_real, real_labels)

            # Train discriminator on fake images
            noise = generate_noise(batch_size, latent_size)
            fake_images: torch.Tensor = generator(noise)
            output_fake = discriminator(fake_images.detach()).view(-1)
            d_loss_fake = criterion(output_fake, fake_labels)

            # Total discriminator loss
            d_loss: torch.Tensor = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train generator to fool discriminator
            optimizer_g.zero_grad()
            output = discriminator(fake_images).view(-1)
            g_loss: torch.Tensor = criterion(output, real_labels)  # We want the generator to output labels as real (1)
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}')

        # Save the best generator model
        if g_loss.item() < best_g_loss:
            best_g_loss = g_loss.item()
            torch.save(generator.state_dict(), best_generator_path)
            print(f"New best generator model saved with loss {best_g_loss:.4f}")

    # Save the final generator model
    torch.save(generator.state_dict(), 'generator_dcgan.pth')


if __name__ == "__main__":
    # Data transformation for CIFAR-10
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models
    generator = Generator(ngpu=NGPU, nz=LATENT_SIZE, ngf=NGF, nc=NC).to(device)
    discriminator = Discriminator(ngpu=NGPU, nc=NC, ndf=NDF).to(device)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # Run training
    train(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, NUM_EPOCHS, LATENT_SIZE)
