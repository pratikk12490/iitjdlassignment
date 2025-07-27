# # contractive_autoencoder.py

# from random import random
# import sys
# import os

# # Ensure the current directory is prioritized in the module search path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import numpy as np
# from torchvision.utils import make_grid
# from math import log10
# import torch.nn.functional as F

# # Hyperparameters
# batch_size = 128
# learning_rate = 1e-3
# num_epochs = 2
# embedding_dim = 64

# # Transform
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])


# # # Dataset
# train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
# test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform, download=True)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# # # Datasets
# # train_dataset = torchvision.datasets.MNIST(
# #     root=local_mnist_path, train=True, transform=transform, download=True
# # )
# # test_dataset = torchvision.datasets.MNIST(
# #     root=local_mnist_path, train=False, transform=transform, download=False
# # )

# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Encoder
# class Encoder(nn.Module):
#     def __init__(self, latent_dim=embedding_dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
#         self.fc = nn.Linear(64 * 7 * 7, latent_dim)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

# # Decoder
# class Decoder(nn.Module):
#     def __init__(self, latent_dim=embedding_dim):
#         super().__init__()
#         self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
#         self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

#     def forward(self, h):
#         x = self.fc(h)
#         x = x.view(-1, 64, 7, 7)
#         x = F.relu(self.deconv1(x))
#         return torch.sigmoid(self.deconv2(x))

# # Contractive Loss
# def contractive_loss(x, x_hat, h, encoder, lam=1e-4):
#     mse = F.mse_loss(x_hat, x)
#     dh = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
#     contractive_penalty = lam * torch.sum(dh ** 2)
#     return mse + contractive_penalty

# # Model and Optimizer
# encoder = Encoder()
# decoder = Decoder()
# optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# # Training Loop
# encoder.train()
# decoder.train()

# for epoch in range(num_epochs):
#     total_loss = 0
#     for x, _ in train_loader:
#         x = x.to(torch.float32)
#         x.requires_grad_()  # ✅ Critical for contractive loss

#         optimizer.zero_grad()
#         h = encoder(x)
#         x_hat = decoder(h)
#         loss = contractive_loss(x, x_hat, h, encoder)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# # t-SNE Visualization
# encoder.eval()
# all_embeddings = []
# all_labels = []

# with torch.no_grad():
#     for x, y in test_loader:
#         h = encoder(x.to(torch.float32))
#         all_embeddings.append(h)
#         all_labels.append(y)

# embeddings = torch.cat(all_embeddings).numpy()
# labels = torch.cat(all_labels).numpy()

# tsne = TSNE(n_components=2, random_state=42)
# emb_2d = tsne.fit_transform(embeddings)

# plt.figure(figsize=(10, 8))
# for digit in range(10):
#     idxs = labels == digit
#     plt.scatter(emb_2d[idxs, 0], emb_2d[idxs, 1], label=str(digit), alpha=0.5)
# plt.legend()
# plt.title("t-SNE Visualization of Contractive Autoencoder Embeddings")
# plt.savefig("contractive_autoencoder/contractive_autoencoder_testing_output/contractive_tsne.png")
# plt.show()



import os
import time
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# Ensure the current directory is prioritized in the module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
embedding_dim = 64
output_dir = 'contractive_autoencoder/contractive_autoencoder_testing_output'

# Set device and random seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)


# Define the Encoder
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

# Define the Decoder
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

# Contractive Loss Function
def contractive_loss(x, x_hat, h, encoder, lam=1e-4):
    mse = F.mse_loss(x_hat, x)
    dh = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
    contractive_penalty = lam * torch.sum(dh ** 2)
    return mse + contractive_penalty

# Training Function
def train_model(encoder, decoder, train_loader, optimizer, num_epochs):
    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(torch.float32)
            x.requires_grad_()  # Critical for contractive loss

            optimizer.zero_grad()
            h = encoder(x)
            x_hat = decoder(h)
            loss = contractive_loss(x, x_hat, h, encoder)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Visualization Function
def visualize_embeddings(encoder, test_loader):
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
    plt.savefig(output_dir + "/contractive_tsne.png")
    plt.show()

def interpolation_experiment(encoder, decoder, test_loader, num_pairs=5):
    """
    Perform interpolation experiment using Contractive Autoencoder
    """
    encoder.eval()
    decoder.eval()
    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print(f"Starting Interpolation Experiment for {num_pairs} pairs...")
    
    # Get test data
    test_data, test_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            test_data.append(data)
            test_labels.append(labels)
            if len(test_data) * data.size(0) >= 1000:
                break
    
    test_data = torch.cat(test_data, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    all_psnr_results = []
    all_l2_results = []
    
    for pair_idx in range(num_pairs):
        print(f"Processing pair {pair_idx + 1}/{num_pairs}")
        
        # Select two images from different classes
        available_classes = torch.unique(test_labels)
        class1, class2 = torch.randperm(len(available_classes))[:2]
        
        # Get indices for each class
        indices_class1 = torch.where(test_labels == available_classes[class1])[0]
        indices_class2 = torch.where(test_labels == available_classes[class2])[0]
        
        # Randomly select one image from each class
        idx1 = indices_class1[torch.randint(0, len(indices_class1), (1,))].item()
        idx2 = indices_class2[torch.randint(0, len(indices_class2), (1,))].item()
        
        I1 = test_data[idx1:idx1+1].to(device)
        I2 = test_data[idx2:idx2+1].to(device)
        
        # Get embeddings h1 and h2
        with torch.no_grad():
            h1 = encoder(I1)
            h2 = encoder(I2)
        
        pair_psnr = []
        pair_l2 = []
        
        # Create visualization for this pair
        fig, axes = plt.subplots(2, len(alpha_values), figsize=(18, 6))
        
        for i, alpha in enumerate(alpha_values):
            # Create interpolated image Iα = αI1 + (1-α)I2
            I_alpha = alpha * I1 + (1 - alpha) * I2
            
            # Get embedding hα = E(Iα)
            with torch.no_grad():
                h_alpha = encoder(I_alpha)
            
            # Get approximate embedding h'α = αh1 + (1-α)h2
            h_prime_alpha = alpha * h1 + (1 - alpha) * h2
            
            # Get reconstructions
            with torch.no_grad():
                I_hat_alpha = decoder(h_alpha)
                I_prime_hat_alpha = decoder(h_prime_alpha)
            
            # Calculate PSNR between Îα and Î'α
            mse = F.mse_loss(I_hat_alpha, I_prime_hat_alpha)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            pair_psnr.append(psnr.item())
            
            # Calculate L2 distance ||hα - h'α||2
            l2_distance = torch.norm(h_alpha - h_prime_alpha, p=2)
            pair_l2.append(l2_distance.item())
            
            # Plot reconstructions
            axes[0, i].imshow(I_hat_alpha[0, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f'I_hat_alpha (alpha={alpha})')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(I_prime_hat_alpha[0, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f"I_prime_hat_alpha (alpha={alpha})")
            axes[1, i].axis('off')
        
        plt.suptitle(f'Pair {pair_idx + 1}: Class {available_classes[class1].item()} vs Class {available_classes[class2].item()}')
        plt.tight_layout()
        interpolation_path = os.path.join(output_dir, f'interpolation_pair_{pair_idx + 1}.png')
        plt.savefig(interpolation_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        all_psnr_results.append(pair_psnr)
        all_l2_results.append(pair_l2)
        
        # Print results for this pair
        print(f"PSNR: {[f'{p:.2f}' for p in pair_psnr]}")
        print(f"L2 Distance: {[f'{l:.4f}' for l in pair_l2]}")
    
    # Calculate and print summary statistics
    print(f"Interpolation Experiment Summary:")
    avg_psnr = np.mean(all_psnr_results, axis=0)
    avg_l2 = np.mean(all_l2_results, axis=0)
    
    print(f"Average PSNR across {num_pairs} pairs:")
    for i, alpha in enumerate(alpha_values):
        print(f"  alpha={alpha}: {avg_psnr[i]:.2f} dB")
    
    print(f"Average L2 Distance across {num_pairs} pairs:")
    for i, alpha in enumerate(alpha_values):
        print(f"  alpha={alpha}: {avg_l2[i]:.4f}")
    
    return all_psnr_results, all_l2_results


# Main Function
def main():
    print("Contractive Autoencoder Implementation")
    print("Deep Learning Assignment")
    
    # Ensure output directory exists
    output_dir = 'contractive_autoencoder_testing_output'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    start_time = time.time()
    # Data Preparation
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model and Optimizer
    encoder = Encoder()
    decoder = Decoder()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Train the Model
    print("Creating Contractive Autoencoder...")

    train_model(encoder, decoder, train_loader, optimizer, num_epochs)

    # Visualize Embeddings
    print("t-SNE Visualization...")
    visualize_embeddings(encoder, test_loader)

    # Perform Interpolation Experiment
    print("Interpolation Experiment")
    interpolation_experiment(encoder, decoder, test_loader, num_pairs=5)

    # Execution Summary
    total_time = time.time() - start_time
    print("Execution completed successfully!")
    print(f"Total Execution Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("Model and results saved in 'contractive_autoencoder/contractive_autoencoder_testing_output/' directory.")

if __name__ == "__main__":
    main()