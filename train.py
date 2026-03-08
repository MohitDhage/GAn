import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import os
import glob

print("SCRIPT START", flush=True)

# Import our models
from inference import ImageEncoder, Generator, LATENT_DIM, IMAGE_FEATURE_DIM
from models_extra import Discriminator
from dataset_pix3d import Pix3DDataset

# Training Parameters
BATCH_SIZE = 8  # Increased from 4 as requested by user
LEARNING_RATE = 0.0002 # Standard for DCGAN stability
BETA1 = 0.5
NUM_EPOCHS = 100 # Higher epochs on smaller dataset for refinement
SAVE_INTERVAL = 1 # Save every epoch to survive future crashes
CHECKPOINT_DIR = Path("checkpoints")
DATASET_ROOT = "dataset"

def get_latest_epoch():
    files = glob.glob(str(CHECKPOINT_DIR / "encoder_epoch_*.pth"))
    if not files:
        return 0
    epochs = [int(f.split("_")[-1].split(".")[0]) for f in files]
    return max(epochs)

def train():
    # Use CUDA if available, but handle the "GPU lost" scenario
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Starting training on GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not found or GPU crashed. Falling back to CPU.", flush=True)
        print("If you just had a 'GPU lost' error, please REBOOT your computer.", flush=True)
    
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # 1. Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        TARGET_CATEGORIES = ['chair', 'sofa', 'table', 'bed']
        print(f"Loading dataset from {DATASET_ROOT} (Mixing categories: {TARGET_CATEGORIES})...", flush=True)
        dataset = Pix3DDataset(DATASET_ROOT, transform=transform, categories=TARGET_CATEGORIES)
        print(f"Dataset loaded. Creating DataLoader with batch size {BATCH_SIZE}...", flush=True)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True) # num_workers=0 is more stable on Windows
    except Exception as e:
        print(f"Error loading dataset: {e}", flush=True)
        return

    # 2. Initialize Models
    encoder = ImageEncoder(out_dim=IMAGE_FEATURE_DIM).to(device)
    generator = Generator(latent_dim=LATENT_DIM, img_feat_dim=IMAGE_FEATURE_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    # 3. Optimizers
    optimizer_G = optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()), 
        lr=LEARNING_RATE, betas=(BETA1, 0.999)
    )
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    # 4. Resume logic
    start_epoch = get_latest_epoch()
    if start_epoch > 0:
        print(f"Resuming from Epoch {start_epoch}...", flush=True)
        encoder.load_state_dict(torch.load(CHECKPOINT_DIR / f"encoder_epoch_{start_epoch}.pth", map_location=device))
        generator.load_state_dict(torch.load(CHECKPOINT_DIR / f"generator_epoch_{start_epoch}.pth", map_location=device))
        # Optional: load discriminator and optimizers too if we want perfect state
    
    criterion = nn.BCELoss()
    real_label = 0.9 # One-sided label smoothing for GAN stability
    fake_label = 0.0
    
    print("Beginning Training Loop...", flush=True)
    
    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        for i, (images, real_voxels) in enumerate(dataloader):
            try:
                batch_size = images.size(0)
                images = images.to(device)
                real_voxels = real_voxels.to(device)
                
                # --- Update Discriminator ---
                discriminator.zero_grad()
                output = discriminator(real_voxels)
                label = torch.full((batch_size, 1), real_label, device=device)
                loss_D_real = criterion(output, label)
                loss_D_real.backward()
                
                noise = torch.randn(batch_size, LATENT_DIM, device=device)
                image_features = encoder(images)
                fake_voxels = generator(image_features, noise)
                
                label.fill_(fake_label)
                output = discriminator(fake_voxels.detach())
                loss_D_fake = criterion(output, label)
                loss_D_fake.backward()
                
                loss_D = loss_D_real + loss_D_fake
                optimizer_D.step()
                
                # --- Update Generator & Encoder ---
                encoder.zero_grad()
                generator.zero_grad()
                label.fill_(real_label)
                output = discriminator(fake_voxels)
                loss_G = criterion(output, label)
                loss_G.backward()
                optimizer_G.step()
                
                if i % 10 == 0:
                    print(f"[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}", flush=True)
                
                # VRAM Cleanup
                if i % 5 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("WARNING: Out of Memory detected. Clearing cache and skipping batch.", flush=True)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Save checkpoints
        if epoch % SAVE_INTERVAL == 0:
            torch.save(encoder.state_dict(), CHECKPOINT_DIR / f"encoder_epoch_{epoch}.pth")
            torch.save(generator.state_dict(), CHECKPOINT_DIR / f"generator_epoch_{epoch}.pth")
            print(f"Saved checkpoints for epoch {epoch}", flush=True)
            
        # Learning Rate decay (Optional but helpful)
        if epoch % 30 == 0:
            for param_group in optimizer_G.param_groups:
                param_group['lr'] *= 0.1
            for param_group in optimizer_D.param_groups:
                param_group['lr'] *= 0.1
            print(f"Decreased LR to {optimizer_G.param_groups[0]['lr']}", flush=True)
            
        print(f"Epoch {epoch} finished in {time.time() - epoch_start:.2f}s", flush=True)

if __name__ == "__main__":
    train()
