import torch
from pathlib import Path
from PIL import Path as PIL_Path
import os
import random

# Reuse our models and inference logic
from inference import run_inference, load_models, LATENT_DIM

def generate_demo_gallery(num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating demo gallery on {device}...")
    
    # 1. Path to common chairs in the dataset for seed images
    dataset_img_dir = Path("dataset/img/chair")
    if not dataset_img_dir.exists():
        print("Dataset images not found. Place them in dataset/img/chair/ to run this.")
        return
        
    img_files = list(dataset_img_dir.glob("*.jpg")) + list(dataset_img_dir.glob("*.png"))
    if not img_files:
        print("No chair images found.")
        return
        
    # 2. Re-triggering inference for random samples
    from torchvision import transforms
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    print(f"Selecting {num_samples} random images from {len(img_files)} chairs...")
    samples = random.sample(img_files, min(num_samples, len(img_files)))
    
    # Ensure outputs directory for the gallery
    gallery_dir = Path("gallery_submission")
    gallery_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(samples):
        print(f"Processing sample {i+1}/{num_samples}: {img_path.name}")
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        # Save a copy of the input image for the gallery
        image.save(gallery_dir / f"sample_{i+1}_input.png")
        
        # Run inference logic using our existing run_inference
        # This will export the .glb to /outputs/ and return metadata
        result = run_inference(
            job_id=f"demo_sample_{i+1}", 
            input_tensor=input_tensor,
            mc_resolution=128, # High res for submission
            export_format="glb"
        )
        
        # Move the generated GLB to the gallery folder for easy organization
        src = Path(f"outputs/demo_sample_{i+1}.glb")
        if src.exists():
            dest = gallery_dir / f"sample_{i+1}_reconstructed.glb"
            if dest.exists(): dest.unlink()
            src.rename(dest)
            print(f"  -> Generated {dest.name}")

    print("\nDEMO GALLERY COMPLETE!")
    print(f"All outputs saved to: {gallery_dir.absolute()}")
    print("Each input image now has a corresponding 3D .glb reconstruction.")

if __name__ == "__main__":
    generate_demo_gallery()
