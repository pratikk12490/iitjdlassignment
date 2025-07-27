import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import os

# Create output directory for testing results
output_dir = 'sparse_autoencoder_testing_output'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

# Set device and random seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# U-Net Architecture without skip connections

class UNetBlock(nn.Module):
    """Basic U-Net block without skip connections"""
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
    """U-Net style encoder WITHOUT skip connections"""
    def __init__(self, in_channels=1, embedding_dim=128):
        super(UNetEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Encoder blocks (no skip connections returned)
        self.enc1 = UNetBlock(in_channels, 64, down=True)    # 28x28 -> 14x14
        self.enc2 = UNetBlock(64, 128, down=True)            # 14x14 -> 7x7
        self.enc3 = UNetBlock(128, 256, down=True)           # 7x7 -> 3x3
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x):
        # Encoder path without storing skip connections
        x1 = self.enc1(x)      # 64x14x14
        x2 = self.enc2(x1)     # 128x7x7
        x3 = self.enc3(x2)     # 256x3x3
        
        embedding = self.bottleneck(x3)
        return embedding

class UNetDecoder(nn.Module):
    """U-Net style decoder WITHOUT skip connections"""
    def __init__(self, embedding_dim=128, out_channels=1):
        super(UNetDecoder, self).__init__()
        
        # Reshape embedding to feature map
        self.reshape = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 3 * 3),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks without skip connections
        self.dec3 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0, output_padding=1)  # 3x3 -> 7x7
        self.dec3_conv = UNetBlock(256, 256, down=False)
        
        self.dec2 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0, output_padding=0)  # 7x7 -> 14x14
        self.dec2_conv = UNetBlock(128, 128, down=False)
        
        self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, output_padding=0)   # 14x14 -> 28x28
        self.dec1_conv = UNetBlock(64, 64, down=False)
        
        # Final output
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, embedding):
        # Reshape embedding
        x = self.reshape(embedding)
        x = x.view(-1, 512, 3, 3)
        
        # Decoder without skip connections
        x = self.dec3(x)        # 3x3 -> 7x7
        x = self.dec3_conv(x)
        
        x = self.dec2(x)        # 7x7 -> 14x14
        x = self.dec2_conv(x)
        
        x = self.dec1(x)        # 14x14 -> 28x28
        x = self.dec1_conv(x)
        
        x = self.final(x)
        return torch.sigmoid(x)

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with U-Net architecture (no skip connections)"""
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
    
    def sparsity_loss(self, embedding):
        """KL divergence sparsity loss"""
        rho = torch.mean(torch.abs(embedding), dim=0)
        rho_hat = self.sparsity_target
        kl_div = rho_hat * torch.log(rho_hat / (rho + 1e-8)) + \
                 (1 - rho_hat) * torch.log((1 - rho_hat) / (1 - rho + 1e-8))
        return torch.sum(kl_div)

# Data loading functions

def load_mnist_data():
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

# Training function

def train_sparse_autoencoder(model, train_loader, num_epochs=5, lr=0.001):
    """Train sparse autoencoder"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            reconstructed, embedding = model(data)
            recon_loss = criterion(reconstructed, data)
            sparsity_loss = model.sparsity_loss(embedding)
            loss = recon_loss + model.sparsity_weight * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_sparsity_loss = total_sparsity_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 1 == 0:
            print(f'Epoch [{epoch+1:2d}/{num_epochs}] - Total: {avg_loss:.4f}, '
                  f'Recon: {avg_recon_loss:.4f}, Sparsity: {avg_sparsity_loss:.4f}')
    
    print("Training completed!")
    return losses

# Analysis functions

def get_embeddings(model, data_loader, max_samples=5000):
    """Extract embeddings from trained model"""
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
    """Plot t-SNE visualization"""
    print(f"Computing t-SNE for {title}...")
    
    # Use a subset for faster computation
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

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def interpolation_experiment(model, test_loader, num_pairs=5):
    """
    Perform interpolation experiment as specified
    """
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
            _, h1 = model(I1)
            _, h2 = model(I2)
        
        pair_psnr = []
        pair_l2 = []
        
        # Create visualization for this pair
        fig, axes = plt.subplots(2, len(alpha_values), figsize=(18, 6))
        
        for i, alpha in enumerate(alpha_values):
            # Create interpolated image Iα = αI1 + (1-α)I2
            I_alpha = alpha * I1 + (1 - alpha) * I2
            
            # Get embedding hα = E(Iα)
            with torch.no_grad():
                _, h_alpha = model(I_alpha)
            
            # Get approximate embedding h'α = αh1 + (1-α)h2
            h_prime_alpha = alpha * h1 + (1 - alpha) * h2
            
            # Get reconstructions
            with torch.no_grad():
                I_hat_alpha = model.decoder(h_alpha)
                I_prime_hat_alpha = model.decoder(h_prime_alpha)
            
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

def classification_accuracy(embeddings, labels):
    """Evaluate classification accuracy using embeddings"""
    print("Computing Classification Accuracy...")
    
    # Split into train/test for classification
    split_idx = int(0.8 * len(embeddings))
    train_embeddings = embeddings[:split_idx]
    train_labels = labels[:split_idx]
    test_embeddings = embeddings[split_idx:]
    test_labels = labels[split_idx:]
    
    # Train classifier
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(train_embeddings, train_labels)
    
    # Predict and compute accuracy
    predictions = classifier.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)
    
    print(f"Classification Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Using {len(train_embeddings)} training and {len(test_embeddings)} test samples")
    
    return accuracy

def plot_training_curve(losses):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Sparse Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

# Main execution

def main():
    print("Sparse Autoencoder Implementation")
    print("Deep Learning Assignment")
    
    start_time = time.time()
    
    # Load Data
    print("Loading MNIST Dataset...")
    train_loader, test_loader = load_mnist_data()
    
    # Create and Train Sparse Autoencoder
    print("Creating Sparse Autoencoder...")
    sparse_ae = SparseAutoencoder(
        embedding_dim=128, 
        sparsity_target=0.05, 
        sparsity_weight=0.001
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in sparse_ae.parameters()):,}")
    
    # Train the model
    train_losses = train_sparse_autoencoder(sparse_ae, train_loader, num_epochs=5)
    
    # t-SNE Visualization
    print("t-SNE Visualization...")
    embeddings, labels = get_embeddings(sparse_ae, test_loader)
    tsne_path = os.path.join(output_dir, "sparse_ae_tsne.png")
    plot_tsne(embeddings, labels, title="Sparse Autoencoder", save_path=tsne_path)
    
    # Interpolation Experiment
    print("Interpolation Experiment...")
    psnr_results, l2_results = interpolation_experiment(sparse_ae, test_loader, num_pairs=5)
    
    # Classification Accuracy
    print("Classification Accuracy...")
    accuracy = classification_accuracy(embeddings, labels)
    
    # Save Model
    print("Saving Model...")
    model_path = os.path.join(output_dir, 'sparse_autoencoder.pth')
    torch.save(sparse_ae.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")
    
    # Plot training curve
    plot_training_curve(train_losses)
    
    # Execution Summary
    total_time = time.time() - start_time
    print("Execution completed successfully!")
    print(f"Total Execution Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Final Classification Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()
