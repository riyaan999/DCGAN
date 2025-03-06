# Deep Convolutional GAN (DCGAN) on CelebA

## Overview
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** based on the paper **"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"** by Radford et al. (2015). The model is trained on the **CelebA dataset** to generate realistic human face images.

## Dataset
- **CelebA Dataset**: A large-scale dataset containing celebrity face images.
- Images are resized to **64×64** and normalized to stabilize training.
- The dataset is loaded using **PyTorch DataLoader** for efficient processing.

## Model Architecture
### 1. Generator (G)
- Takes a **random noise vector (100-dimensional)** as input.
- Uses **ConvTranspose2D, BatchNorm, and ReLU** layers to generate a realistic image.
- Outputs a **64×64 RGB image** with values in the range **[-1, 1]** using a Tanh activation.

### 2. Discriminator (D)
- Takes a **64×64 RGB image** as input (real or generated).
- Uses **Convolutional layers with LeakyReLU** for feature extraction.
- Outputs a single probability value indicating whether the image is real or fake using a Sigmoid activation.

## Training Details
- **Binary Cross-Entropy Loss (BCELoss)** is used for both Generator and Discriminator.
- **Adam Optimizer** is used with a learning rate of `0.0002` and `betas (0.5, 0.999)`.
- The model is trained for **10 epochs** to speed up execution.
- Batch size is set to **128** for stable training.

## Results
- The generator progressively improves and learns to create human-like faces.
- Loss curves for both Generator and Discriminator are plotted to track training stability.
- A set of generated images is displayed at the end of training.

## Requirements
To run this project, install the following dependencies:

```bash
pip install torch torchvision matplotlib tqdm
