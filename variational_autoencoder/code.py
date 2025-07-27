# Variational Autoencoder

# 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from scipy.io import loadmat
from IPython.display import clear_output
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# 2. Load dataset
def load_frey_faces():
    print("\n=== Loading Frey Faces Dataset ===")
    try:
        print("Attempting to load frey_rawface.mat...")
        data = loadmat('frey_rawface.mat')
        
        if 'ff' in data:
            print("Found 'ff' variable in MAT file")
            images = data['ff'].T.reshape(-1, 28, 20)  # (1965, 28, 20)
        elif 'frey_rawface' in data:
            print("Found 'frey_rawface' variable in MAT file")
            images = data['frey_rawface'].reshape(-1, 28, 20)
        else:
            raise ValueError("MAT file doesn't contain expected variables")
        
        print(f"Raw images shape: {images.shape}")
        images = images.astype('float32') / 255.0
        images = images[:, np.newaxis, :, :]  # (1965, 1, 28, 20)
        print(f"Normalized images shape: {images.shape}")
        print(f"Pixel value range: {images.min():.2f} to {images.max():.2f}")
        
        return images

    except Exception as e:
        print(f"\nError loading MAT file: {e}")
        print("Possible causes:")
        print("1. File not found in current directory")
        print("2. File is corrupted")
        print("3. File doesn't contain expected variables ('ff' or 'frey_rawface')")
        
        print("\nTrying to upload file...")
        try:
            from google.colab import files
            uploaded = files.upload()
            if 'frey_rawface.mat' not in uploaded:
                raise FileNotFoundError("Could not find frey_rawface.mat after upload")
            print("File uploaded successfully, retrying load...")
            return load_frey_faces()  # Try again after upload
        except ImportError:
            print("Not running in Colab, cannot use files.upload()")
            raise FileNotFoundError("Could not find frey_rawface.mat")

class FreyFaceDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()
        print(f"\nDataset initialized with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 3. Enhanced VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=(1, 28, 20), latent_dim=20):
        super(VAE, self).__init__()
        print(f"\n=== Initializing VAE Model ===")
        print(f"Input dimension: {input_dim}")
        print(f"Latent dimension: {latent_dim}")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened dimension
        with torch.no_grad():
            sample = torch.zeros(1, *input_dim)
            conv_out = self.encoder(sample)
            self.flatten_dim = conv_out.shape[1]
            print(f"Encoder output dimension: {conv_out.shape}")
            print(f"Flatten dimension: {self.flatten_dim}")

        # Latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)
        print(f"Mu layer: {self.fc_mu}")
        print(f"Logvar layer: {self.fc_var}")

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        print(f"Decoder input layer: {self.decoder_input}")

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 7, 5)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        # Print decoder architecture
        print("\nDecoder architecture:")
        for i, layer in enumerate(self.decoder):
            print(f"Layer {i+1}: {layer}")

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# 4. Enhanced Training Function 
def train_vae(model, dataloader, epochs=50, lr=1e-3):
    print("\n=== Training VAE ===")
    print(f"Training parameters:")
    print(f"- Epochs: {epochs}")
    print(f"- Learning rate: {lr}")
    print(f"- Batch size: {dataloader.batch_size}")
    print(f"- Training samples: {len(dataloader.dataset)}")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        reconstruction_loss = 0
        kl_loss = 0

        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(data)
            
            # Calculate losses
            BCE = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = BCE + KLD

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            reconstruction_loss += BCE.item()
            kl_loss += KLD.item()

        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon = reconstruction_loss / len(dataloader.dataset)
        avg_kl = kl_loss / len(dataloader.dataset)
        train_losses.append((avg_loss, avg_recon, avg_kl))

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_vae_model.pth')
            print(f"New best model saved with loss: {best_loss:.2f}")

        # Detailed epoch logging
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"- Total Loss: {avg_loss:.2f}")
        print(f"- Reconstruction Loss: {avg_recon:.2f} ({(avg_recon/avg_loss)*100:.1f}% of total)")
        print(f"- KL Divergence: {avg_kl:.2f} ({(avg_kl/avg_loss)*100:.1f}% of total)")
        
        # Log latent space statistics
        if epoch % 5 == 0 or epoch == epochs-1:
            with torch.no_grad():
                sample = next(iter(dataloader))[:1].to(device)
                _, mu, log_var = model(sample)
                print(f"\nLatent space stats for sample:")
                print(f"- Mean range: {mu.min().item():.2f} to {mu.max().item():.2f}")
                print(f"- Logvar range: {log_var.min().item():.2f} to {log_var.max().item():.2f}")
                std_dev = torch.exp(0.5*log_var)
                print(f"- Effective latent dimensions (std > 0.5): {(std_dev > 0.5).sum().item()}/{model.latent_dim}")

        # Visualization
        if epoch % 5 == 0 or epoch == epochs-1:
            clear_output(wait=True)
            plt.figure(figsize=(15, 5))

            # Loss plots
            plt.subplot(1, 3, 1)
            plt.plot([x[0] for x in train_losses])
            plt.title(f"Total Loss\nEpoch {epoch+1}/{epochs}")
            plt.xlabel('Epoch')
            
            plt.subplot(1, 3, 2)
            plt.plot([x[1] for x in train_losses])
            plt.title("Reconstruction Loss")
            plt.xlabel('Epoch')
            
            plt.subplot(1, 3, 3)
            plt.plot([x[2] for x in train_losses])
            plt.title("KL Divergence")
            plt.xlabel('Epoch')

            plt.tight_layout()
            plt.show()

            # Reconstruction samples
            with torch.no_grad():
                sample = next(iter(dataloader))[:5].to(device)
                recon, _, _ = model(sample)
                plt.figure(figsize=(10, 4))
                plt.imshow(make_grid(torch.cat([sample, recon]).cpu(), nrow=5).permute(1, 2, 0))
                plt.axis('off')
                plt.title("Original (top) vs Reconstructed (bottom)")
                plt.show()

    return model, train_losses

# 5. Enhanced Evaluation and Visualization Functions
def evaluate(model, dataloader):
    print("\n=== Model Evaluation ===")
    model.eval()
    test_loss = 0
    reconstruction_loss = 0
    kl_loss = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            
            # Calculate losses separately
            BCE = F.binary_cross_entropy(recon_batch, data, reduction='sum').item()
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()).item()
            
            test_loss += BCE + KLD
            reconstruction_loss += BCE
            kl_loss += KLD
    
    test_loss /= len(dataloader.dataset)
    reconstruction_loss /= len(dataloader.dataset)
    kl_loss /= len(dataloader.dataset)
    
    print(f"Evaluation Results:")
    print(f"- Total Loss: {test_loss:.2f}")
    print(f"- Reconstruction Loss: {reconstruction_loss:.2f} ({(reconstruction_loss/test_loss)*100:.1f}% of total)")
    print(f"- KL Divergence: {kl_loss:.2f} ({(kl_loss/test_loss)*100:.1f}% of total)")

def visualize_latent_space(model, latent_dim=20):
    print("\n=== Visualizing Latent Space ===")
    print(f"Generating visualizations for each of {latent_dim} latent dimensions...")
    model.eval()
    with torch.no_grad():
        # Create figure for each latent dimension
        for dim in range(latent_dim):
            plt.figure(figsize=(20, 2))
            plt.suptitle(f'Latent Dimension {dim} Variation', y=1.05)
            
            # Linearly spaced coordinates along the standard normal
            grid_x = np.linspace(-3, 3, 10)
            
            # Generate images by varying one latent dimension
            for i, xi in enumerate(grid_x):
                z = torch.zeros(1, latent_dim).to(device)
                z[0, dim] = xi
                
                sample = model.decode(z).cpu()
                plt.subplot(1, 10, i+1)
                plt.imshow(sample[0, 0], cmap='gray')
                plt.axis('off')
                plt.title(f'{xi:.1f}')
            
            plt.show()
            print(f"Visualized dimension {dim+1}/{latent_dim}")

def generate_samples(model, n_samples=10, latent_dim=20):
    print(f"\n=== Generating {n_samples} Random Samples ===")
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        print(f"Sampled latent vectors shape: {z.shape}")
        print(f"Latent vector stats - Mean: {z.mean().item():.2f}, Std: {z.std().item():.2f}")
        
        samples = model.decode(z).cpu()
        
        plt.figure(figsize=(20, 2))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i+1)
            plt.imshow(samples[i, 0], cmap='gray')
            plt.axis('off')
        plt.suptitle('Randomly Generated Faces', y=1.05)
        plt.show()

def interpolate(model, n_steps=10, latent_dim=20):
    print(f"\n=== Latent Space Interpolation with {n_steps} steps ===")
    model.eval()
    with torch.no_grad():
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)
        
        print(f"Start latent vector: {z1[0,:5].cpu().numpy()}...")  # Print first 5 dims
        print(f"End latent vector: {z2[0,:5].cpu().numpy()}...")
        
        plt.figure(figsize=(20, 2))
        for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
            z = alpha * z1 + (1 - alpha) * z2
            sample = model.decode(z).cpu()
            
            plt.subplot(1, n_steps, i+1)
            plt.imshow(sample[0, 0], cmap='gray')
            plt.axis('off')
            plt.title(f'{alpha:.1f}')
        
        plt.suptitle('Latent Space Interpolation', y=1.05)
        plt.show()

# 6. Main Execution with enhanced error handling
if __name__ == '__main__':
    print("=== Frey Faces VAE Implementation ===")
    try:
        # Load data
        print("\n[Step 1/6] Loading data...")
        images = load_frey_faces()
        print(f"Successfully loaded {len(images)} images of shape {images.shape[1:]}")

        dataset = FreyFaceDataset(images)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Show samples
        print("\n[Step 2/6] Displaying sample images...")
        plt.figure(figsize=(10, 2))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(images[i, 0], cmap='gray')
            plt.axis('off')
        plt.show()

        # Initialize model
        print("\n[Step 3/6] Initializing VAE model...")
        latent_dim = 20
        vae = VAE(latent_dim=latent_dim)
        print(f"VAE model initialized with {sum(p.numel() for p in vae.parameters()):,} parameters")
        print(f"Model architecture:\n{vae}")

        # Train model
        print("\n[Step 4/6] Training model...")
        vae, losses = train_vae(vae, dataloader, epochs=50)

        # Evaluation
        print("\n[Step 5/6] Evaluating model...")
        evaluate(vae, dataloader)

        # Visualization
        print("\n[Step 6/6] Running visualizations...")
        print("\nVisualizing latent space variations:")
        visualize_latent_space(vae)

        print("\nGenerating random samples:")
        generate_samples(vae)

        print("\nShowing latent space interpolation:")
        interpolate(vae)

        print("\n=== Training Complete ===")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nDebugging information:")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'args'):
            print(f"Error arguments: {e.args}")
        
        print("\nTroubleshooting steps:")
        print("1. Ensure frey_rawface.mat is in your working directory")
        print("2. Verify the file contains either 'ff' or 'frey_rawface' variable")
        print("3. Check that you have sufficient GPU memory if using CUDA")
        print("4. Try reducing batch size if you encounter memory errors")
        print("5. Verify all required packages are installed")
        
