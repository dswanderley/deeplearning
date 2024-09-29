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


# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize to [-1, 1]
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Loss and optimizers
criterion = nn.BCELoss()
generator = Generator(LATENT_SIZE, HIDDEN_SIZE, IMAGE_SIZE)
discriminator = Discriminator(IMAGE_SIZE, HIDDEN_SIZE)

optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# Function to generate random noise
def generate_noise(batch_size, latent_size):
    return torch.randn(batch_size, latent_size)

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (images, _) in enumerate(train_loader):
        BATCH_SIZE = images.size(0)
        images = images.view(BATCH_SIZE, -1)  # Flatten the image to [batch_size, 784]

        # Create real and fake labels
        real_labels = torch.ones(BATCH_SIZE, 1)
        fake_labels = torch.zeros(BATCH_SIZE, 1)

        # Train discriminator on real images
        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)

        # Train discriminator on fake images
        noise = generate_noise(BATCH_SIZE, LATENT_SIZE)
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

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}')

# Save final model
torch.save(generator.state_dict(), 'generator.pth')

