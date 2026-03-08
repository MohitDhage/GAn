import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path

class Pix3DDataset(Dataset):
    def __init__(self, root_dir, transform=None, voxel_res=64, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images and pix3d.json.
            transform (callable, optional): Optional transform to be applied on a sample.
            voxel_res (int): Target resolution for voxels.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.voxel_res = voxel_res
        
        json_path = self.root_dir / "pix3d.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Could not find pix3d.json in {root_dir}")
            
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Filter only entries that have voxels and (optionally) match specified categories
        self.entries = [e for e in self.metadata if 'voxel' in e]
        
        target_categories = kwargs.get('categories')
        if target_categories:
            if isinstance(target_categories, str):
                target_categories = [target_categories]
            self.entries = [e for e in self.entries if e.get('category') in target_categories]
             
        print(f"Loaded Pix3D dataset with {len(self.entries)} entries (Categories: {target_categories if target_categories else 'All'}).")

    def __len__(self):
        return len(self.entries)

    def load_binvox(self, path):
        """
        Simple binvox reader.
        """
        with open(path, 'rb') as f:
            line = f.readline().strip()
            if not line.startswith(b'#binvox'):
                raise IOError('Not a binvox file')
            
            dims = f.readline().split()[1:4]
            dims = [int(d) for d in dims]
            
            f.readline() # translate
            f.readline() # scale
            
            line = f.readline().strip()
            if not line.startswith(b'data'):
                raise IOError('Data marker not found')
            
            raw_data = np.frombuffer(f.read(), dtype=np.uint8)
            values, counts = raw_data[::2], raw_data[1::2]
            
            data = np.zeros(np.prod(dims), dtype=bool)
            current = 0
            for v, c in zip(values, counts):
                data[current:current+c] = v
                current += c
                
            return data.reshape(dims)

    def load_voxel(self, path):
        """
        Supports both .mat and .binvox files.
        """
        if path.suffix == ".mat":
            from scipy.io import loadmat
            mat = loadmat(path)
            # Pix3D .mat files usually contain 'voxel' or 'vol' or similar
            if 'voxel' in mat:
                return mat['voxel']
            elif 'vol' in mat:
                return mat['vol']
            else:
                # Fallback: find the first 3D array
                for key, val in mat.items():
                    if isinstance(val, np.ndarray) and val.ndim == 3:
                        return val
                raise IOError(f"Could not find voxel data in {path}")
        else:
            return self.load_binvox(path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.entries[idx]
        
        # Load image
        img_path = self.root_dir / entry['img']
        image = Image.open(img_path).convert('RGBA').convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Load voxel
        voxel_path = self.root_dir / entry['voxel']
        voxels = self.load_voxel(voxel_path)
        
        if voxels.shape[0] != self.voxel_res:
            from scipy.ndimage import zoom
            scale = self.voxel_res / voxels.shape[0]
            voxels = zoom(voxels.astype(float), zoom=scale, order=0) > 0.5
            
        voxels = torch.from_numpy(voxels.astype(np.float32)).unsqueeze(0) # (1, D, H, W)
        
        return image, voxels

if __name__ == "__main__":
    # Test dataset (mock root)
    try:
        ds = Pix3DDataset("dataset/pix3d")
        print(f"Dataset size: {len(ds)}")
    except Exception as e:
        print(f"Dataset test failed (expected if data not yet downloaded): {e}")
