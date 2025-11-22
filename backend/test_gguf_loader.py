"""Test GGUF model loading and generation"""
from app.utils.gguf_loader import GGUFLoader
import os

print("🧪 Testing GGUF Model Loading...\n")

# Model path
MODEL_PATH = "data/models/quantized/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("   Run: python download_test_model.py")
    exit(1)

print(f"✅ Model found: {MODEL_PATH}")
model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"✅ Size: {model_size:.2f} MB\n")

# Test 1: Load model
print("1️⃣ Testing model loading...")
loader = GGUFLoader(model_path=MODEL_PATH)

try:
    loader.load()
    print("   ✅ Model loaded successfully!")
except Exception as e:
    print(f"   ❌ Failed to load: {e}")
    exit(1)

# Test 2: Generate text
print("\n2️⃣ Testing text generation...")
result = loader.generate(
    prompt="What is artificial intelligence? Answer briefly.",
    max_tokens=50
)

if result["success"]:
    print(f"   ✅ Generation successful!")
    print(f"   📝 Output: {result['output_text'][:100]}...")
    print(f"   ⏱️  Latency: {result['latency_ms']:.2f}ms")
    print(f"   🔢 Tokens: {result['token_count']}")
    print(f"   🚀 Speed: {result['tokens_per_second']:.2f} tok/s")
else:
    print(f"   ❌ Generation failed: {result['error']}")
    exit(1)

# Test 3: Batch generation
print("\n3️⃣ Testing batch generation (3 prompts)...")
prompts = [
    "What is machine learning?",
    "What is deep learning?",
    "What is a neural network?"
]

for i, prompt in enumerate(prompts, 1):
    result = loader.generate(prompt, max_tokens=30)
    if result["success"]:
        print(f"   ✅ Prompt {i}: {result['latency_ms']:.0f}ms, {result['token_count']} tokens")
    else:
        print(f"   ❌ Prompt {i} failed")

# Test 4: Unload model
print("\n4️⃣ Testing model unloading...")
loader.unload()
print("   ✅ Model unloaded")

print("\n" + "="*50)
print("🎉 GGUF LOADER TEST COMPLETE!")
print("="*50)
print("\n✅ GGUF loading working!")
print("✅ Text generation working!")
print("✅ Ready for quantized generation task!")