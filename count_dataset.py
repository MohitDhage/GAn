import json
from pathlib import Path

DATASET_ROOT = r"d:\3d_gan_project\dataset"
json_path = Path(DATASET_ROOT) / "pix3d.json"

if not json_path.exists():
    print(f"Could not find pix3d.json in {DATASET_ROOT}")
else:
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    entries = [e for e in metadata if 'voxel' in e]
    print(f"Total entries: {len(metadata)}")
    print(f"Entries with voxels: {len(entries)}")
