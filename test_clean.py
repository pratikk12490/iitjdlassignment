"""
Test Script for Trained Sparse Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = "sparse_autoencoder_testing_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
else:
    print(f"Using existing output directory: {output_dir}")

# Model architecture (same as training)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super(UNetBlock, self).__init__()
        self.down = down
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.pool = nn.MaxPool2d(2)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        if self.down:
            conv_out = self.conv(x)
            pooled = self.pool(conv_out)
            return pooled
        else:
            return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=128):
        super(UNetEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.enc1 = UNetBlock(in_channels, 64, down=True)
        self.enc2 = UNetBlock(64, 128, down=True)
        self.enc3 = UNetBlock(128, 256, down=True)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        embedding = self.bottleneck(x3)
        return embedding

class UNetDecoder(nn.Module):
    def __init__(self, embedding_dim=128, out_channels=1):
        super(UNetDecoder, self).__init__()
        
        self.reshape = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 3 * 3),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0, output_padding=1)
        self.dec3_conv = UNetBlock(256, 256, down=False)
        
        self.dec2 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0, output_padding=0)
        self.dec2_conv = UNetBlock(128, 128, down=False)
        
        self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, output_padding=0)
        self.dec1_conv = UNetBlock(64, 64, down=False)
        
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, embedding):
        x = self.reshape(embedding)
        x = x.view(-1, 512, 3, 3)
        
        x = self.dec3(x)
        x = self.dec3_conv(x)
        
        x = self.dec2(x)
        x = self.dec2_conv(x)
        
        x = self.dec1(x)
        x = self.dec1_conv(x)
        
        x = self.final(x)
        return torch.sigmoid(x)

class SparseAutoencoder(nn.Module):
    def __init__(self, embedding_dim=128, sparsity_weight=0.001, sparsity_target=0.05):
        super(SparseAutoencoder, self).__init__()
        self.encoder = UNetEncoder(embedding_dim=embedding_dim)
        self.decoder = UNetDecoder(embedding_dim=embedding_dim)
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        
    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return reconstructed, embedding

# Data loading

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"MNIST test data loaded: {len(test_dataset)} samples")
    return test_loader

# Analysis functions

def get_embeddings(model, data_loader, max_samples=5000):
    model.eval()
    embeddings = []
    labels = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if len(embeddings) * data_loader.batch_size >= max_samples:
                break
                
            data = data.to(device)
            embedding = model.encoder(data)
            
            embeddings.append(embedding.cpu().numpy())
            labels.append(target.numpy())
    
    embeddings_array = np.vstack(embeddings)
    labels_array = np.hstack(labels)
    print(f"Extracted {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
    return embeddings_array, labels_array

def plot_tsne(embeddings, labels, title="Sparse Autoencoder", save_path=None):
    print(f"Computing t-SNE for {title}...")
    
    n_samples = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    embeddings_subset = embeddings[indices]
    labels_subset = labels[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_subset)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels_subset, cmap='tab10', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Digit Class')
    plt.title(f't-SNE Visualization - {title}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    
    plt.show()
    print("t-SNE visualization completed!")

def interpolation_experiment(model, test_loader, num_pairs=5):
    model.eval()
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
        
        indices_class1 = torch.where(test_labels == available_classes[class1])[0]
        indices_class2 = torch.where(test_labels == available_classes[class2])[0]
        
        idx1 = indices_class1[torch.randint(0, len(indices_class1), (1,))].item()
        idx2 = indices_class2[torch.randint(0, len(indices_class2), (1,))].item()
        
        I1 = test_data[idx1:idx1+1].to(device)
        I2 = test_data[idx2:idx2+1].to(device)
        
        with torch.no_grad():
            _, h1 = model(I1)
            _, h2 = model(I2)
        
        pair_psnr = []
        pair_l2 = []
        
        # Visualization
        fig, axes = plt.subplots(2, len(alpha_values), figsize=(18, 6))
        
        for i, alpha in enumerate(alpha_values):
            I_alpha = alpha * I1 + (1 - alpha) * I2
            
            with torch.no_grad():
                _, h_alpha = model(I_alpha)
            
            h_prime_alpha = alpha * h1 + (1 - alpha) * h2
            
            with torch.no_grad():
                I_hat_alpha = model.decoder(h_alpha)
                I_prime_hat_alpha = model.decoder(h_prime_alpha)
            
            # Calculate metrics
            mse = F.mse_loss(I_hat_alpha, I_prime_hat_alpha)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            pair_psnr.append(psnr.item())
            
            l2_distance = torch.norm(h_alpha - h_prime_alpha, p=2)
            pair_l2.append(l2_distance.item())
            
            # Plot
            axes[0, i].imshow(I_hat_alpha[0, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f'I_hat_alpha (alpha={alpha})')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(I_prime_hat_alpha[0, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f"I_prime_hat_alpha (alpha={alpha})")
            axes[1, i].axis('off')
        
        plt.suptitle(f'Pair {pair_idx + 1}: Class {available_classes[class1].item()} vs Class {available_classes[class2].item()}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/test_interpolation_pair_{pair_idx + 1}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        all_psnr_results.append(pair_psnr)
        all_l2_results.append(pair_l2)
        
        print(f"PSNR: {[f'{p:.2f}' for p in pair_psnr]}")
        print(f"L2 Distance: {[f'{l:.4f}' for l in pair_l2]}")
    
    return all_psnr_results, all_l2_results

def classification_accuracy(embeddings, labels):
    print("Evaluating classification accuracy...")
    
    split_idx = int(0.8 * len(embeddings))
    X_train, X_test = embeddings[:split_idx], embeddings[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy

# Load trained model

def load_trained_model(model_path='sparse_autoencoder.pth'):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        print("Please train the model first by running: python train_only.py")
        return None
    
    model = SparseAutoencoder(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded trained model from '{model_path}'")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def main():
    print("Testing Trained Sparse Autoencoder")
    
    start_time = time.time()
    
    # Load trained model
    print("Loading trained model...")
    model = load_trained_model()
    if model is None:
        return
    
    # Load test data
    print("Loading MNIST test data...")
    test_loader = load_mnist_data()
    
    # Extract embeddings and t-SNE
    print("Extracting embeddings and generating t-SNE...")
    embeddings, labels = get_embeddings(model, test_loader)
    plot_tsne(embeddings, labels, "Trained Sparse Autoencoder", save_path=f"{output_dir}/test_tsne.png")
    
    # Interpolation experiment
    print("Performing interpolation experiments...")
    psnr_results, l2_results = interpolation_experiment(model, test_loader, num_pairs=5)
    
    # Classification accuracy
    print("Evaluating classification accuracy...")
    accuracy = classification_accuracy(embeddings, labels)
    
    # Summary
    total_time = time.time() - start_time
    print("Testing completed!")
    print(f"Total Testing Time: {total_time:.1f} seconds")
    print(f"Classification Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Visualizations saved in '{output_dir}/' folder: test_tsne.png, test_interpolation_pair_*.png")

if __name__ == "__main__":
    main()
