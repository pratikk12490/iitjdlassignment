# Sparse Autoencoder Implementation

This folder contains the complete implementation of the Sparse Autoencoder for the Deep Learning assignment.

## Files

### Core Implementation
- **`main.py`** - Complete implementation with training and testing (all-in-one)
- **`train_clean.py`** - Training script only (saves model as sparse_autoencoder.pth)
- **`test_clean.py`** - Testing script only (loads trained model and runs analysis)

### Generated Files
- **`sparse_autoencoder.pth`** - Trained model weights (generated after training)
- **`sparse_autoencoder_testing_output/`** - Output folder containing:
  - `test_tsne.png` - t-SNE visualization
  - `test_interpolation_pair_*.png` - Interpolation experiment results

## Usage

### Option 1: Run complete implementation
```bash
cd sparse_autoencoder
python main.py
```

### Option 2: Separate training and testing
```bash
cd sparse_autoencoder

# Step 1: Train model
python train_clean.py

# Step 2: Test model  
python test_clean.py
```

## Assignment Requirements Implemented

1. **Sparse Autoencoder** with U-Net architecture (no skip connections)
2. **t-SNE visualization** with ground truth class labels  
3. **Interpolation experiments** with α values [0, 0.2, 0.4, 0.6, 0.8, 1.0]
4. **PSNR and L2 distance** calculations
5. **Classification accuracy** using learned embeddings

## Model Architecture

- **Encoder**: U-Net style encoder without skip connections
- **Decoder**: U-Net style decoder without skip connections  
- **Embedding Dimension**: 128
- **Sparsity Loss**: KL divergence regularization
- **Dataset**: MNIST (28x28 grayscale digits)

## Results

The model achieves approximately 89% classification accuracy using the learned embeddings with a simple logistic regression classifier.
