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
BATCH_SIZE = 8
LEARNING_RATE_G = 0.0001   # Slightly lower for generator stability
LEARNING_RATE_D = 0.0004   # TTUR: train discriminator faster
BETA1 = 0.5
NUM_EPOCHS = 200
SAVE_INTERVAL = 1
CHECKPOINT_DIR = Path("checkpoints_v2")   # Fresh directory for new training regime
DATASET_ROOT = "dataset"

# Loss weights — reconstruction loss is critical to prevent mode collapse
LAMBDA_RECON = 10.0   # L1 reconstruction weight (high to force shape learning)
LAMBDA_ADV   = 1.0    # Adversarial weight

def get_latest_epoch():
    files = glob.glob(str(CHECKPOINT_DIR / "encoder_epoch_*.pth"))
    if not files:
        return 0
    epochs = [int(f.split("_")[-1].split(".")[0]) for f in files]
    return max(epochs)

def train():
    # Use CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Starting training on GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not found. Falling back to CPU.", flush=True)
    
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # 1. Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        TARGET_CATEGORIES = ['chair', 'sofa', 'table', 'bed']
        print(f"Loading dataset from {DATASET_ROOT} (Categories: {TARGET_CATEGORIES})...", flush=True)
        dataset = Pix3DDataset(DATASET_ROOT, transform=transform, categories=TARGET_CATEGORIES)
        print(f"Dataset loaded: {len(dataset)} samples. Batch size: {BATCH_SIZE}", flush=True)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    except Exception as e:
        print(f"Error loading dataset: {e}", flush=True)
        return

    # 2. Initialize Models
    encoder = ImageEncoder(out_dim=IMAGE_FEATURE_DIM).to(device)
    generator = Generator(latent_dim=LATENT_DIM, img_feat_dim=IMAGE_FEATURE_DIM).to(device)
    discriminator = Discriminator().to(device)
    
    # 3. Optimizers (TTUR — Two Time-scale Update Rule)
    optimizer_G = optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()), 
        lr=LEARNING_RATE_G, betas=(BETA1, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), 
        lr=LEARNING_RATE_D, betas=(BETA1, 0.999)
    )
    
    # 4. Resume logic
    start_epoch = get_latest_epoch()
    if start_epoch > 0:
        print(f"Resuming from Epoch {start_epoch}...", flush=True)
        encoder.load_state_dict(torch.load(CHECKPOINT_DIR / f"encoder_epoch_{start_epoch}.pth", map_location=device))
        generator.load_state_dict(torch.load(CHECKPOINT_DIR / f"generator_epoch_{start_epoch}.pth", map_location=device))
        disc_path = CHECKPOINT_DIR / f"discriminator_epoch_{start_epoch}.pth"
        if disc_path.exists():
            discriminator.load_state_dict(torch.load(disc_path, map_location=device))
    
    # Loss functions
    criterion_adv = nn.BCELoss()        # Adversarial loss
    criterion_recon = nn.L1Loss()       # Reconstruction loss (L1 = sharper than L2)
    real_label_val = 0.9  # One-sided label smoothing
    fake_label_val = 0.0
    
    print(f"Training: λ_recon={LAMBDA_RECON}, λ_adv={LAMBDA_ADV}", flush=True)
    print(f"LR_G={LEARNING_RATE_G}, LR_D={LEARNING_RATE_D}", flush=True)
    print("Beginning Training Loop...", flush=True)
    
    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_loss_recon = 0.0
        n_batches = 0
        
        for i, (images, real_voxels) in enumerate(dataloader):
            try:
                batch_size = images.size(0)
                images = images.to(device)
                real_voxels = real_voxels.to(device)
                
                # ============================================================
                # UPDATE DISCRIMINATOR
                # ============================================================
                discriminator.zero_grad()
                
                # Real voxels
                output_real = discriminator(real_voxels)
                label_real = torch.full((batch_size, 1), real_label_val, device=device)
                loss_D_real = criterion_adv(output_real, label_real)
                loss_D_real.backward()
                
                # Fake voxels
                noise = torch.randn(batch_size, LATENT_DIM, device=device)
                image_features = encoder(images)
                fake_voxels = generator(image_features, noise)
                
                label_fake = torch.full((batch_size, 1), fake_label_val, device=device)
                output_fake = discriminator(fake_voxels.detach())
                loss_D_fake = criterion_adv(output_fake, label_fake)
                loss_D_fake.backward()
                
                loss_D = loss_D_real + loss_D_fake
                optimizer_D.step()
                
                # ============================================================
                # UPDATE GENERATOR + ENCODER
                # ============================================================
                encoder.zero_grad()
                generator.zero_grad()
                
                # Adversarial loss: fool the discriminator
                output_fake_for_G = discriminator(fake_voxels)
                label_real_for_G = torch.full((batch_size, 1), real_label_val, device=device)
                loss_G_adv = criterion_adv(output_fake_for_G, label_real_for_G)
                
                # *** RECONSTRUCTION LOSS — the key fix ***
                # Forces the generator to actually match ground-truth voxels
                loss_recon = criterion_recon(fake_voxels, real_voxels)
                
                # Combined generator loss
                loss_G = LAMBDA_ADV * loss_G_adv + LAMBDA_RECON * loss_recon
                loss_G.backward()
                optimizer_G.step()
                
                # Track metrics
                epoch_loss_D += loss_D.item()
                epoch_loss_G += loss_G_adv.item()
                epoch_loss_recon += loss_recon.item()
                n_batches += 1
                
                if i % 10 == 0:
                    # Also log occupancy stats of fake voxels for monitoring
                    with torch.no_grad():
                        fake_occ = (fake_voxels > 0.5).float().mean().item() * 100
                        real_occ = (real_voxels > 0.5).float().mean().item() * 100
                    print(
                        f"[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}] "
                        f"D: {loss_D.item():.4f} | G_adv: {loss_G_adv.item():.4f} | "
                        f"Recon: {loss_recon.item():.4f} | "
                        f"Occ(fake/real): {fake_occ:.1f}%/{real_occ:.1f}%",
                        flush=True
                    )
                
                # VRAM Cleanup
                if i % 5 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("WARNING: OOM. Clearing cache and skipping batch.", flush=True)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Epoch summary
        if n_batches > 0:
            avg_D = epoch_loss_D / n_batches
            avg_G = epoch_loss_G / n_batches
            avg_R = epoch_loss_recon / n_batches
        else:
            avg_D = avg_G = avg_R = 0.0
        
        # Save checkpoints (encoder, generator, AND discriminator)
        if epoch % SAVE_INTERVAL == 0:
            torch.save(encoder.state_dict(), CHECKPOINT_DIR / f"encoder_epoch_{epoch}.pth")
            torch.save(generator.state_dict(), CHECKPOINT_DIR / f"generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), CHECKPOINT_DIR / f"discriminator_epoch_{epoch}.pth")
            
        # Learning Rate decay — gentler schedule
        if epoch % 50 == 0:
            for param_group in optimizer_G.param_groups:
                param_group['lr'] *= 0.5
            for param_group in optimizer_D.param_groups:
                param_group['lr'] *= 0.5
            print(f"LR halved → G: {optimizer_G.param_groups[0]['lr']:.6f}, D: {optimizer_D.param_groups[0]['lr']:.6f}", flush=True)
            
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch} ({elapsed:.1f}s) | "
            f"Avg D: {avg_D:.4f} | Avg G_adv: {avg_G:.4f} | Avg Recon: {avg_R:.4f}",
            flush=True
        )

if __name__ == "__main__":
    train()

