"""
Diagnostic script to check what the trained GAN generator is actually producing.
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from inference import ImageEncoder, Generator, load_models, LATENT_DIM, IMAGE_FEATURE_DIM
from pathlib import Path
import glob

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 2. Load models (will autodiscover latest checkpoint)
encoder, generator = load_models(device)

# 3. Load a sample image from the training set
sample_images = list(Path("dataset/img/chair").rglob("*.png"))[:3]
if not sample_images:
    sample_images = list(Path("dataset/img/bed").rglob("*.png"))[:3]

print(f"\nTesting with {len(sample_images)} sample images from training data...")

# Use the same transform as training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for img_path in sample_images:
    print(f"\n--- Image: {img_path.name} ---")
    
    # Load and preprocess
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 256, 256)
    print(f"  Input tensor shape: {img_tensor.shape}")
    print(f"  Input tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
    
    # Run through encoder + generator
    with torch.no_grad():
        image_features = encoder(img_tensor)
        print(f"  Image features shape: {image_features.shape}")
        print(f"  Image features range: [{image_features.min():.4f}, {image_features.max():.4f}]")
        
        noise = torch.randn(1, LATENT_DIM, device=device)
        scalar_field = generator(image_features, noise)
        print(f"  Generator output shape: {scalar_field.shape}")
        
        sf_np = scalar_field[0, 0].cpu().numpy()
        print(f"  Scalar field range: [{sf_np.min():.6f}, {sf_np.max():.6f}]")
        print(f"  Scalar field mean:  {sf_np.mean():.6f}")
        print(f"  Scalar field std:   {sf_np.std():.6f}")
        
        # Check occupancy at various thresholds
        for threshold in [0.5, 0.3, 0.1, 0.05, 0.01]:
            occupied = (sf_np >= threshold).sum()
            total = sf_np.size
            pct = occupied / total * 100
            print(f"  Voxels >= {threshold}: {occupied}/{total} ({pct:.2f}%)")

# 4. Also test with raw (unnormalized) image to compare
print("\n\n=== COMPARISON: Without normalization ===")
img = Image.open(sample_images[0]).convert("RGB")
img_raw = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])(img).unsqueeze(0).to(device)

with torch.no_grad():
    feat = encoder(img_raw)
    noise = torch.randn(1, LATENT_DIM, device=device)
    sf = generator(feat, noise)
    sf_np = sf[0, 0].cpu().numpy()
    print(f"  Scalar field range (no norm): [{sf_np.min():.6f}, {sf_np.max():.6f}]")
    print(f"  Scalar field mean (no norm):  {sf_np.mean():.6f}")
    for threshold in [0.5, 0.3, 0.1, 0.05]:
        occupied = (sf_np >= threshold).sum()
        total = sf_np.size
        print(f"  Voxels >= {threshold}: {occupied}/{total} ({occupied/total*100:.2f}%)")

# 5. Check what a perfect random noise gives
print("\n\n=== Random Image (Sanity Check) ===")
random_img = torch.randn(1, 3, 256, 256, device=device)
with torch.no_grad():
    feat = encoder(random_img)
    noise = torch.randn(1, LATENT_DIM, device=device)
    sf = generator(feat, noise)
    sf_np = sf[0, 0].cpu().numpy()
    print(f"  Scalar field range (random): [{sf_np.min():.6f}, {sf_np.max():.6f}]")
    print(f"  Scalar field mean (random):  {sf_np.mean():.6f}")
    for threshold in [0.5, 0.3, 0.1]:
        occupied = (sf_np >= threshold).sum()
        total = sf_np.size
        print(f"  Voxels >= {threshold}: {occupied}/{total} ({occupied/total*100:.2f}%)")

print("\n=== Diagnosis Complete ===")
