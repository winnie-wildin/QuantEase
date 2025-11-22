"""
Groq API Client - Wrapper for baseline model generation
Uses Groq's cloud API for fast inference without local GPU
"""
import os
from groq import Groq
from typing import Optional, Dict, Any
import time


class GroqClient:
    """
    Wrapper for Groq API to generate baseline outputs.
    
    Supports multiple models from Groq's API including:
    - llama-3.3-70b-versatile (recommended - newest!)
    - llama-3.1-70b-versatile (good balance)
    - mixtral-8x7b-32768 (long context)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (or uses GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.default_model = "llama-3.3-70b-versatile"
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Groq API.
        
        Args:
            prompt: Input text to generate from
            model: Model to use (defaults to llama-3.3-70b-versatile)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters for Groq API
        
        Returns:
            Dict containing:
                - output_text: Generated text
                - latency_ms: Generation time in milliseconds
                - token_count: Number of tokens generated
                - tokens_per_second: Generation speed
                - model_used: Which model was used
        """
        model = model or self.default_model
        
        # Start timing
        start_time = time.time()
        
        try:
            # Call Groq API (non-streaming)
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                **kwargs
            )
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            output_text = response.choices[0].message.content
            
            # Get token counts from response
            token_count = len(output_text.split())  # Rough estimate
            if hasattr(response, 'usage') and response.usage:
                token_count = response.usage.completion_tokens
            
            tokens_per_second = token_count / (latency_ms / 1000.0) if latency_ms > 0 else 0
            
            return {
                "output_text": output_text,
                "latency_ms": latency_ms,
                "token_count": token_count,
                "tokens_per_second": tokens_per_second,
                "model_used": model,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "output_text": "",
                "latency_ms": latency_ms,
                "token_count": 0,
                "tokens_per_second": 0,
                "model_used": model,
                "success": False,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test if Groq API connection works.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = self.generate("Say 'Hello!'", max_tokens=10)
            return result["success"]
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


# Convenience function for quick usage
def generate_with_groq(prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Quick function to generate text with Groq.
    
    Args:
        prompt: Input text
        **kwargs: Additional parameters (model, max_tokens, etc.)
    
    Returns:
        Generation result dictionary
    """
    client = GroqClient()
    return client.generate(prompt, **kwargs)