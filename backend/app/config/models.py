"""
Model Configuration - Available baseline and quantized models
"""

GROQ_MODELS = [
    {
        "id": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B",
        "description": "Meta's flagship model - excellent quality",
        "provider": "groq",
        "speed": "⚡ Very Fast",
        "quality": "🏆 Excellent"
    },
    {
        "id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B",
        "description": "Smaller, faster Llama model",
        "provider": "groq",
        "speed": "⚡⚡ Ultra Fast",
        "quality": "⭐ Good"
    },
    {
        "id": "qwen/qwen3-32b",
        "name": "Qwen 3 32B",
        "description": "Alibaba's latest model - multilingual",
        "provider": "groq",
        "speed": "⚡ Very Fast",
        "quality": "🏆 Excellent"
    },
    {
        "id": "moonshotai/kimi-k2-instruct-0905",
        "name": "Kimi K2 Instruct",
        "description": "Moonshot AI's Kimi model - Chinese focused",
        "provider": "groq",
        "speed": "⚡ Very Fast",
        "quality": "🏆 Excellent"
    }
]
# Available quantized models (GGUF) - must be pre-downloaded
QUANTIZED_MODELS = [
    {
        "id": "tinyllama-1.1b",
        "name": "TinyLlama 1.1B (INT4)",
        "description": "Smallest model - very fast on CPU",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_mb": 638,
        "quantization": "INT4",
        "speed": "🐌 Slow (CPU)",
        "quality": "⚠️ Basic"
    },
    {
        "id": "qwen2.5-3b",
        "name": "Qwen 2.5 3B (INT4)",
        "description": "Alibaba's small model - good quality",
        "filename": "qwen2.5-3b-instruct-q4.gguf",
        "size_mb": 1900,
        "quantization": "INT4",
        "speed": "🐌 Slow (CPU)",
        "quality": "⭐ Good"
    },
    {
        "id": "phi3-mini",
        "name": "Phi-3 Mini 3.8B (INT4)",
        "description": "Microsoft's efficient model",
        "filename": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "size_mb": 2300,
        "quantization": "INT4",
        "speed": "🐌 Slow (CPU)",
        "quality": "⭐ Good"
    }
]


def get_groq_model(model_id: str):
    """Get Groq model config by ID"""
    return next((m for m in GROQ_MODELS if m["id"] == model_id), None)


def get_quantized_model(model_id: str):
    """Get quantized model config by ID"""
    return next((m for m in QUANTIZED_MODELS if m["id"] == model_id), None)