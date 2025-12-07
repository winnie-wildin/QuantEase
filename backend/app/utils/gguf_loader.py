#app/utils/gguf_loader.py
"""
GGUF Model Loader - Loads and runs quantized GGUF models locally.
Uses llama-cpp-python for fast CPU/GPU inference with quantized models.
"""
import os
from typing import Optional, Dict, Any
from llama_cpp import Llama
import time
from app.utils.prompt_formatter import PromptFormatter  # ADD THIS


class GGUFLoader:
    """
    Loader for GGUF quantized models.
    
    Supports various quantization levels:
    - Q8_0: 8-bit quantization (~7GB for 7B models)
    - Q4_K_M: 4-bit quantization (~4GB for 7B models)
    - Q4_0: 4-bit quantization, smaller (~3.5GB for 7B models)
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0
    ):
        """
        Initialize GGUF loader.
        
        Args:
            model_path: Path to .gguf model file
            n_ctx: Context window size (max tokens)
            n_threads: Number of CPU threads
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        self.is_loaded = False
        
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Get model size
        self.model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    def load(self):
        """
        Load the GGUF model into memory.
        
        Returns:
            Llama model instance
        """
        if self.is_loaded:
            return self.model
        
        print(f"Loading GGUF model from {self.model_path}...")
        print(f"Model size: {self.model_size_mb:.2f} MB")
        
        start_time = time.time()
        
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            
            load_time = time.time() - start_time
            self.is_loaded = True
            
            print(f"✅ Model loaded in {load_time:.2f}s")
            return self.model
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        max_retries: int = 3
    ) -> dict:
        """
        Generate text with the loaded model.
        Auto-formats prompt and uses model-specific stop sequences.
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Format prompt for this model
        formatted_prompt = PromptFormatter.format_prompt(prompt, self.model_path)
        stop_sequences = PromptFormatter.get_stop_sequences(self.model_path)
        
        for attempt in range(max_retries):
            start_time = time.time()
            
            try:
                print(f"🔍 DEBUG: Calling llama-cpp-python with max_tokens={max_tokens}")
                print(f"🔍 DEBUG: Stop sequences: {stop_sequences}")
                # Generate with model-specific settings
                output = self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=40,
                    repeat_penalty=1.15,  # Prevent repetition
                    echo=False,
                    stop=stop_sequences  # Model-specific stops
                )
                # Add this IMMEDIATELY after generation
                actual_tokens = output["usage"]["completion_tokens"]
                print(f"🔍 DEBUG: Model returned {actual_tokens} tokens (requested max: {max_tokens})")
                print(f"🔍 DEBUG: Finish reason: {output['choices'][0].get('finish_reason')}")
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # Extract text
                generated_text = output["choices"][0]["text"].strip()
                
                # Remove any stop sequences that leaked through
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0].strip()
                
                # Validate output
                if generated_text and len(generated_text) >= 5:
                    # Success!
                    token_count = output["usage"]["completion_tokens"]
                    tokens_per_second = token_count / (latency_ms / 1000) if latency_ms > 0 else 0
                    
                    return {
                        "output_text": generated_text,
                        "latency_ms": latency_ms,
                        "token_count": token_count,
                        "tokens_per_second": tokens_per_second,
                        "model_path": self.model_path,
                        "success": True,
                        "error": None
                    }
                
                # Empty or too short - retry
                print(f"⚠️  Attempt {attempt + 1}/{max_retries}: Output too short ({len(generated_text)} chars)")
                print(f"    Output: '{generated_text}'")
                
                # Increase temperature for next attempt
                temperature = min(1.0, temperature + 0.2)
                print(f"    Retrying with temperature={temperature}")
                
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        "output_text": "[Generation failed after retries]",
                        "latency_ms": 0,
                        "token_count": 0,
                        "tokens_per_second": 0,
                        "model_path": self.model_path,
                        "success": False,
                        "error": str(e)
                    }
        
        # All retries exhausted
        return {
            "output_text": "[No valid output after 3 attempts]",
            "latency_ms": 0,
            "token_count": 0,
            "tokens_per_second": 0,
            "model_path": self.model_path,
            "success": False,
            "error": "Empty output after all retries"
        }
    def unload(self):
        """
        Unload model from memory to free up RAM.
        """
        if self.is_loaded:
            self.model = None
            self.is_loaded = False
            print(f"✅ Model unloaded from memory")
    
    def __enter__(self):
        """Context manager entry - load model"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model"""
        self.unload()


def get_quantization_from_filename(filename: str) -> str:
    """
    Extract quantization level from GGUF filename.
    
    Args:
        filename: GGUF filename (e.g., "llama-2-7b.Q8_0.gguf")
    
    Returns:
        Quantization level (e.g., "INT8", "INT4")
    """
    filename_lower = filename.lower()
    
    if 'q8' in filename_lower:
        return "INT8"
    elif 'q4' in filename_lower:
        return "INT4"
    elif 'q5' in filename_lower:
        return "INT5"
    elif 'q6' in filename_lower:
        return "INT6"
    elif 'f16' in filename_lower or 'fp16' in filename_lower:
        return "FP16"
    else:
        return "UNKNOWN"