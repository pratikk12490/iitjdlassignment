# Deep Learning Programming Assignment

**Due Date:** July 27, 2025  
**Max Marks:** 100

## Instructions:

- You can make a team of maximum 5 students.
- Prepare a report containing the group members details (with contributions), results and training details.
- Submit a zip file (named as RollNo1_..._RollNo5.zip) with the following structure. **I will not evaluate if it is not in this structure.**

```
- code.py
- report.pdf
- data
  * train/
    - image1
    .
    .
  * test/
    - image1
    .
    .
```

- **Copying** from Internet, GenAI tools, other groups would lead to an **F grade** and the case would be reported to the Institute committee for further actions. I already have the GenAI solutions with me, if your solution matches with those (even with change of variable names) would be considered as copy.

## Assignment Tasks

### 1. Sparse Auto-encoder and Contractive Auto-encoder

Implement the Sparse auto-encoder and the Contractive Auto-encoder. Use the MNIST digit dataset for training images. Use the U-Net auto-encoder architecture for encoder and decoder with proper skip connections. Let **E** be the trained encoder and **D** be the trained decoder, **h = E(I)** be the embedding of an image **I** and **I' = D(h)** be the output of the decoder.

**(a) (20 points)** Plot the t-sne (use inbuilt function) on the embeddings obtained using the respective auto-encoders. Color the clusters using the respective ground truth labels.

**(b) (20 points)** Randomly take two images **I₁** and **I₂** from two different digit classes. Let **h₁ = E(I₁)** and **h₂ = E(I₂)** be the embeddings for these images, respectively. Construct another image **hᵢₙₜ = αh₁ + (1 − α)h₂** for **α ∈ (0.2, 0.4, 0.6, 0.8, 1)**. Find the embeddings **hα**, of this image **Iᵢₙₜ**, for all values of α by passing it through the encoder. Also, consider the approximate embedding **h'α ≈ αh₁ + (1 − α)h₂** by using directly the embeddings of the images **I₁** and **I₂**. Also, find **I'α = D(h'α)** and **I'α = D(hα)**.
Plot the images **Iᵢₙₜ** and **I'α** side by side for different values of α. Do this for 20 pairs (**I₁, I₂**). Report PSNR between **Iᵢₙₜ** and **I'α** for all values of alpha.

**(c) (20 points)** After training the autoencoders, you want to check if the embeddings of different digits are different and embeddings within a class are similar. For this purpose, you propose to perform the classification of the digits based on the embeddings obtained by the encoders and check the accuracy of classifications for each of the Auto-encoder. Report the classification accuracy as well as overall of the AE and report which one is better. Use any inbuilt classifier to solve the classification problem.

### 2. Variational Auto-encoders (40 points)

Implement variational Auto-encoders. Use the Frey Face dataset to train your network. Sample points from the learned distribution by varying different latent variables to show that your network has learned meaningful latent variables. Set the embedding vector size to 20.

---

## Repository Structure

This repository follows the required structure:
- `code.py` - Main implementation file
- `report.pdf` - Assignment report with results and analysis
- `data/` - Dataset folder
  - `train/` - Training images
  - `test/` - Testing images

## Getting Started

1. Clone this repository
2. Install required dependencies
3. Run the code using: `python code.py`
4. Check the results in the generated outputs

## Important Notes

- **Academic Integrity**: This assignment must be completed without copying from internet sources, GenAI tools, or other groups
- **Team Size**: Maximum 5 students per team
- **Submission Format**: Submit as RollNo1_..._RollNo5.zip with exact folder structure
- **Due Date**: July 27, 2025
