"""Test Groq API connection and generation"""
from app.utils.groq_client import GroqClient
import os
from dotenv import load_dotenv
load_dotenv()
print("🧪 Testing Groq API Integration...\n")

# Check API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not found in environment!")
    print("   Add it to your .env file")
    exit(1)

print(f"✅ API Key found: {api_key[:10]}...")

# Test connection
print("\n1️⃣ Testing connection...")
client = GroqClient()
if client.test_connection():
    print("   ✅ Connection successful!")
else:
    print("   ❌ Connection failed!")
    exit(1)

# Test generation
print("\n2️⃣ Testing text generation...")
result = client.generate(
    prompt="What is artificial intelligence? Answer in one sentence.",
    max_tokens=50
)

if result["success"]:
    print(f"   ✅ Generation successful!")
    print(f"   📝 Output: {result['output_text'][:100]}...")
    print(f"   ⏱️  Latency: {result['latency_ms']:.2f}ms")
    print(f"   🔢 Tokens: {result['token_count']}")
    print(f"   🚀 Speed: {result['tokens_per_second']:.2f} tokens/sec")
    print(f"   🤖 Model: {result['model_used']}")
else:
    print(f"   ❌ Generation failed: {result['error']}")
    exit(1)

# Test multiple generations
print("\n3️⃣ Testing batch generation (3 samples)...")
prompts = [
    "What is machine learning?",
    "What is deep learning?",
    "What is neural network?"
]

for i, prompt in enumerate(prompts, 1):
    result = client.generate(prompt, max_tokens=30)
    if result["success"]:
        print(f"   ✅ Sample {i}: {result['latency_ms']:.0f}ms")
    else:
        print(f"   ❌ Sample {i} failed")

print("\n" + "="*50)
print("🎉 PHASE 2 - GROQ API INTEGRATION COMPLETE!")
print("="*50)
print("\n✅ Groq client working!")
print("✅ Ready to generate baseline outputs!")
print("\n🚀 Next: Test baseline generation task")