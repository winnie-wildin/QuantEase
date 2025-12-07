"""
Test script for Groq qwen/qwen3-32b with RAG input
This mimics exactly what baseline_generation.py does
"""
from dotenv import load_dotenv
import os

load_dotenv()
from groq import Groq

# Initialize client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Test with a RAG-style input (similar to your dataset)
question = "What is the primary purpose of a Bloom filter?"
context = """
A Bloom filter is a space-efficient probabilistic data structure used to test 
whether an element is a member of a set. It can return false positives but never 
false negatives. Bloom filters use multiple hash functions to map elements to 
positions in a bit array. They are commonly used in databases and caches to 
quickly check if an item might exist before performing expensive lookups.
"""

# Build RAG prompt (same as TaskPromptBuilder)
rag_prompt = f"""Answer the following question based ONLY on the information provided in the context below. Do not use external knowledge.

Context:
{context}

Question: {question}

Answer:"""

# System message (same as PromptFormatter)
system_message = "Respond directly and concisely. Do not show your thinking process or use <think> tags."

# Stop sequences (same as baseline_generation.py)
stop_sequences = None
print("="*80)
print("🧪 TESTING GROQ MODEL: qwen/qwen3-32b")
print("="*80)
print("\n📋 CONFIGURATION:")
print(f"   Model: qwen/qwen3-32b")
print(f"   System message: {system_message}")
print(f"   Stop sequences: {stop_sequences}")
print("\n📥 INPUT PROMPT:")
print(rag_prompt)
print("\n" + "="*80)
print("🚀 CALLING GROQ API...")
print("="*80 + "\n")

# Make the API call
try:
    completion = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=0.95,
        stop=stop_sequences
    )
    
    # Extract response
    output_text = completion.choices[0].message.content
    
    print("✅ API CALL SUCCESSFUL")
    print("\n📤 RAW OUTPUT:")
    print("-"*80)
    print(repr(output_text))  # Show with quotes to see whitespace
    print("-"*80)
    
    print("\n📊 OUTPUT ANALYSIS:")
    print(f"   Length: {len(output_text)} characters")
    print(f"   Contains '<think>': {('<think>' in output_text)}")
    print(f"   Contains '</think>': {('</think>' in output_text)}")
    print(f"   First 200 chars: {output_text[:200]}")
    
    # Test cleaning logic
    print("\n🧹 TESTING CLEANING LOGIC:")
    print("-"*80)
    
    original_length = len(output_text)
    cleaned = output_text
    
    if '<think>' in output_text:
        print("⚠️  Detected <think> tags!")
        parts = output_text.split('</think>')
        cleaned = parts[-1].strip() if len(parts) > 1 else output_text.split('<think>')[0].strip()
        print(f"   After cleaning: '{cleaned}'")
        print(f"   Cleaned length: {len(cleaned)} chars")
    else:
        print("✅ No <think> tags detected")
    
    if len(cleaned) < 10:
        print(f"\n❌ PROBLEM: Output too short after cleaning ({len(cleaned)} chars)")
        print("\n🔍 DEBUGGING INFO:")
        print(f"   Original output had {original_length} chars")
        print(f"   Split by '</think>': {output_text.split('</think>')}")
        print(f"   Parts[-1]: '{parts[-1] if '</think>' in output_text else 'N/A'}'")
    else:
        print(f"\n✅ SUCCESS: Cleaned output is {len(cleaned)} chars")
        print(f"\n📄 FINAL OUTPUT:")
        print(cleaned)
    
    # Also show completion metadata
    print("\n📈 COMPLETION METADATA:")
    print(f"   Model used: {completion.model}")
    print(f"   Finish reason: {completion.choices[0].finish_reason}")
    if hasattr(completion, 'usage'):
        print(f"   Tokens used: {completion.usage}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("🏁 TEST COMPLETE")
print("="*80)