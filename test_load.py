import torch
import sys
import os
sys.path.append(os.getcwd())
from inference import load_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
try:
    encoder, generator = load_models(device)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
