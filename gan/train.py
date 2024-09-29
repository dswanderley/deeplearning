import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from models import Generator, Discriminator


# Configurations
BATCH_SIZE = 64
LATENT_SIZE = 64  # Size of the noise vector
HIDDEN_SIZE = 256
IMAGE_SIZE = 28 * 28  # MNIST image size (28x28)
NUM_EPOCHS = 50
LEARNING_RATE = 0.0002


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to generate random noise
def generate_noise(batch_size, latent_size):
    return torch.randn(batch_size, latent_size, device=device)


# Training function
def train(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, num_epochs, latent_size):
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            batch_size = images.size(0)
            images = images.view(batch_size, -1).to(device)  # Flatten and move images to GPU if available

            # Create real and fake labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train discriminator on real images
            outputs = discriminator(images)
            d_loss_real = criterion(outputs, real_labels)

            # Train discriminator on fake images
            noise = generate_noise(batch_size, latent_size)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            # Backpropagation for discriminator
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train generator to fool discriminator
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            # Backpropagation for generator
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}')

    # Save the trained generator model
    torch.save(generator.state_dict(), 'generator.pth')


# Main block
if __name__ == "__main__":
    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize to [-1, 1]
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Loss and optimizers
    criterion = nn.BCELoss()
    generator = Generator(LATENT_SIZE, HIDDEN_SIZE, IMAGE_SIZE).to(device)  # Move generator to GPU if available
    discriminator = Discriminator(IMAGE_SIZE, HIDDEN_SIZE).to(device)  # Move discriminator to GPU if available

    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # Run training
    train(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, NUM_EPOCHS, LATENT_SIZE)
