# 3D-GAN Model Documentation: Conditional 3D Generative Adversarial Network

This document provides a detailed technical overview of the **3D-GAN** architecture implemented in this project for **2D-to-3D Asset Generation**.

---

## 1. High-Level Architecture
The system is built as a **Conditional 3D Generative Adversarial Network (3D-cGAN)**. Unlike a standard GAN that generates images from random noise, this model is "conditioned" on a 2D input image to produce a specific 3D shape that matches the visual context of the photo.

### The Three Pillars of the Model:
1.  **Image Encoder (The Observer):** A 2D Convolutional Neural Network (CNN) that processes the input image and extracts a compact "feature vector" representing the object's style, perspective, and category.
2.  **3D Generator (The Artist):** A 3D Transposed Convolutional Network that takes the feature vector and "paints" it into a volumetric voxel grid (64x64x64).
3.  **3D Discriminator (The Critic):** A 3D Convolutional Network that evaluates the generated voxel grid. It compares it against real 3D data from the Pix3D dataset and provides feedback to the Generator to improve its output.

---

## 2. Technical Component Details

### A. Image Encoder (2D CNN)
The encoder is responsible for mapping the high-dimensional image space to a latent feature space.
*   **Architecture:** Multiple layers of 2D convolutions with Batch Normalization and LeakyReLU activations.
*   **Input:** RGB Image $(256 \times 256 \times 3)$.
*   **Output:** Feature Vector (Dimension: 256).

### B. 3D Generator (Volumetric Synthesis)
The Generator performs "inverse convolution" to expand the latent features into a 3D volume.
*   **Latent Space:** Concat(Image Features + Random Noise Vector). Dimension: $256 + 512 = 768$.
*   **Upsampling:** Uses **3D Transposed Convolutions** (`ConvTranspose3d`) to progressively increase spatial resolution from $4 \times 4 \times 4 \rightarrow 8^3 \rightarrow 16^3 \rightarrow 32^3 \rightarrow 64^3$.
*   **Final Layer:** Sigmoid activation, producing an occupancy field where values closer to 1 indicate "solid matter" (voxels) and values closer to 0 indicate "empty space."

### C. 3D Discriminator (Shape Verification)
The Discriminator acts as a classifier that ensures the generated furniture looks structurally sound.
*   **Architecture:** 3D CNN using **LeakyReLU** to handle the sparse nature of 3D voxel data.
*   **Input:** Voxel grid $(64 \times 64 \times 64)$.
*   **Logic:** It learns to recognize the structural patterns of real furniture (e.g., straight legs for chairs, flat surfaces for tables) and penalizes the Generator for producing "floating blobs" or unrealistic noise.

---

## 3. Training Strategy & Stability
GANs are notoriously difficult to train. This project implements several "Deep Convolutional GAN" (DCGAN) best practices to ensure high-quality results within a short timeframe:

1.  **One-Sided Label Smoothing:** Real labels are set to **0.9** instead of 1.0. This prevents the Discriminator from becoming "too confident," which allows the Generator to continue learning even in the later stages of training.
2.  **Adam Optimizers:** Standard learning rate $(\eta = 0.0002)$ with Momentum $(\beta_1 = 0.5)$ for stable adversarial updates.
3.  **Category Focus:** By filtering the dataset (e.g., focusing on chairs, sofas, tables), the model learns class-specific geometry (like four-legged support or backrests) much faster than a universal model.
4.  **Learning Rate Decay:** The learning rate is reduced periodically (every 30 epochs) to allow the model to "settle" and refine fine geometric details.

---

## 4. Inference & Post-Processing
Once the model produces a voxel grid, the following steps convert it into a usable 3D asset:

1.  **Thresholding:** Values $> 0.5$ are treated as solid voxels.
2.  **Marching Cubes:** A surface reconstruction algorithm ($128^3$ resolution) that wraps the voxel grid in a smooth triangular mesh.
3.  **Cleanup:** Automatic removal of "floating artifacts" (disconnected noise) by keeping only the largest connected geometric component.
4.  **Export:** The final mesh is exported as a **.GLB** or **.OBJ** file, which can be opened in Blender, Windows 3D Viewer, or WebGL frontends.

---

## 5. Summary of Model Pipeline
$$ \text{2D Image} \xrightarrow{\text{Encoder}} \text{Features} \xrightarrow{\text{+ Noise}} \text{Generator} \xrightarrow{\text{Voxels}} \text{Marching Cubes} \xrightarrow{\text{Mesh}} \text{3D GLB} $$
