# contractive_autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
embedding_dim = 64
local_mnist_path = '/Users/ashishnathani/Documents/IITJ-Material/DL/archive'

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Datasets
train_dataset = torchvision.datasets.MNIST(
    root=local_mnist_path, train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root=local_mnist_path, train=False, transform=transform, download=False
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=embedding_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def forward(self, h):
        x = self.fc(h)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        return torch.sigmoid(self.deconv2(x))

# Contractive Loss
def contractive_loss(x, x_hat, h, encoder, lam=1e-4):
    mse = F.mse_loss(x_hat, x)
    dh = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
    contractive_penalty = lam * torch.sum(dh ** 2)
    return mse + contractive_penalty

# Model and Optimizer
encoder = Encoder()
decoder = Decoder()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Training Loop
encoder.train()
decoder.train()

for epoch in range(num_epochs):
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(torch.float32)
        x.requires_grad_()  # ✅ Critical for contractive loss

        optimizer.zero_grad()
        h = encoder(x)
        x_hat = decoder(h)
        loss = contractive_loss(x, x_hat, h, encoder)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# t-SNE Visualization
encoder.eval()
all_embeddings = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        h = encoder(x.to(torch.float32))
        all_embeddings.append(h)
        all_labels.append(y)

embeddings = torch.cat(all_embeddings).numpy()
labels = torch.cat(all_labels).numpy()

tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
for digit in range(10):
    idxs = labels == digit
    plt.scatter(emb_2d[idxs, 0], emb_2d[idxs, 1], label=str(digit), alpha=0.5)
plt.legend()
plt.title("t-SNE Visualization of Contractive Autoencoder Embeddings")
plt.savefig("contractive_tsne.png")
plt.show()
