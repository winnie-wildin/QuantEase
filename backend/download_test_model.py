"""Download a small GGUF model for testing"""
import os
import requests
from tqdm import tqdm

# Create directory
os.makedirs("data/models/quantized", exist_ok=True)

# Small test model: TinyLlama 1.1B Q4_K_M (~600MB)
MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = "data/models/quantized/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

print("📥 Downloading TinyLlama 1.1B Q4_K_M (~600MB)...")
print(f"   URL: {MODEL_URL}")
print(f"   Saving to: {MODEL_PATH}")

# Check if already exists
if os.path.exists(MODEL_PATH):
    print(f"✅ Model already exists!")
    print(f"   Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    exit(0)

# Download with progress bar
response = requests.get(MODEL_URL, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(MODEL_PATH, 'wb') as f:
    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

print(f"\n✅ Download complete!")
print(f"   Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
print(f"   Path: {MODEL_PATH}")