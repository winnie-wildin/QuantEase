"""
Prompt Formatter - Model-specific prompt templates
Different models expect different formats for best results
"""
from typing import Dict, List


class PromptFormatter:
    """
    Format prompts for different model families.
    Each model has specific instruction formats and stop sequences.
    """
    
    # Model-specific formats
    FORMATS = {
        "tinyllama": {
            "template": "### Instruction:\n{prompt}\n\n### Response:\n",
            "stop_sequences": ["###", "\n\n\n", "Instruction:", "</s>"],
            "description": "TinyLlama 1.1B - Alpaca instruction format",
            "system_message": None
        },
        
        "llama2": {
            "template": "<s>[INST] {prompt} [/INST]",
            "stop_sequences": ["</s>", "[INST]", "[/INST]"],
            "description": "Llama-2 chat format",
            "system_message": None
        },
        
        "llama3": {
            "template": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "stop_sequences": ["<|eot_id|>", "<|end_of_text|>"],
            "description": "Llama-3 chat format",
            "system_message": None
        },
        
        "mistral": {
            "template": "<s>[INST] {prompt} [/INST]",
            "stop_sequences": ["</s>", "[INST]"],
            "description": "Mistral/Mixtral instruction format",
            "system_message": None
        },
        
        "phi": {
            "template": "Instruct: {prompt}\nOutput:",
            "stop_sequences": ["\nInstruct:", "<|endoftext|>"],
            "description": "Phi-2/3 instruction format",
            "system_message": None
        },
        
        "gemma": {
            "template": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
            "stop_sequences": ["<end_of_turn>", "<start_of_turn>"],
            "description": "Gemma instruction format",
            "system_message": None
        },
        
        "qwen": {
            "template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "stop_sequences": ["<think>", "</think>"],  # ← Only thinking tags, remove <|im_end|>
            "description": "Qwen chat format",
            "system_message": "Respond directly and concisely. Do not show your thinking process or use <think> tags."
        },
        
        "default": {
            "template": "{prompt}",
            "stop_sequences": ["\n\n"],
            "description": "Generic/unknown model - raw prompt",
            "system_message": None
        }
    }
    
    @staticmethod
    def detect_model_family(model_path: str) -> str:
        """
        Detect model family from path/filename.
        
        Args:
            model_path: Path to model file or model name
            
        Returns:
            Model family identifier (e.g., "tinyllama", "llama2")
        """
        model_lower = model_path.lower()
        
        # Check for specific model families
        if "tinyllama" in model_lower or "tiny-llama" in model_lower:
            return "tinyllama"
        elif "llama-3" in model_lower or "llama3" in model_lower:
            return "llama3"
        elif "llama-2" in model_lower or "llama2" in model_lower:
            return "llama2"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        elif "phi-" in model_lower or "phi2" in model_lower or "phi3" in model_lower:
            return "phi"
        elif "gemma" in model_lower:
            return "gemma"
        elif "qwen" in model_lower:
            return "qwen"
        else:
            return "default"
    
    @staticmethod
    def format_prompt(prompt: str, model_path: str) -> str:
        """
        Format prompt for the specific model.
        
        Args:
            prompt: Raw input text
            model_path: Path to model file
            
        Returns:
            Formatted prompt ready for generation
        """
        model_family = PromptFormatter.detect_model_family(model_path)
        format_config = PromptFormatter.FORMATS.get(model_family, PromptFormatter.FORMATS["default"])
        
        formatted = format_config["template"].format(prompt=prompt.strip())
        
        print(f"📋 Model family detected: {model_family}")
        print(f"📝 Format: {format_config['description']}")
        
        return formatted
    
    @staticmethod
    def get_stop_sequences(model_path: str) -> List[str]:
        """
        Get stop sequences for the specific model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            List of stop sequences
        """
        model_family = PromptFormatter.detect_model_family(model_path)
        format_config = PromptFormatter.FORMATS.get(model_family, PromptFormatter.FORMATS["default"])
        
        return format_config["stop_sequences"]
    
    @staticmethod
    def get_system_message(model_name: str):
        """
        Get system message for the specific model.
        
        Args:
            model_name: Model name or path
            
        Returns:
            System message or None
        """
        model_family = PromptFormatter.detect_model_family(model_name)
        format_config = PromptFormatter.FORMATS.get(model_family, PromptFormatter.FORMATS["default"])
        return format_config.get("system_message")
    
    @staticmethod
    def get_stop_sequences_for_api(model_name: str) -> list:
        """
        Get stop sequences for API calls (works with model names, not just paths).
        
        Args:
            model_name: Model name (e.g., "qwen/qwen3-32b")
            
        Returns:
            List of stop sequences
        """
        model_family = PromptFormatter.detect_model_family(model_name)
        format_config = PromptFormatter.FORMATS.get(model_family, PromptFormatter.FORMATS["default"])
        return format_config.get("stop_sequences", [])
    
    @staticmethod
    def add_custom_format(
        name: str,
        template: str,
        stop_sequences: List[str],
        description: str = "",
        system_message: str = None
    ):
        """
        Add a custom prompt format for a new model.
        
        Args:
            name: Model family identifier
            template: Prompt template with {prompt} placeholder
            stop_sequences: List of stop tokens
            description: Human-readable description
            system_message: Optional system message
            
        Example:
            >>> PromptFormatter.add_custom_format(
            ...     name="my_model",
            ...     template="Q: {prompt}\nA:",
            ...     stop_sequences=["\nQ:", "</answer>"],
            ...     description="My custom model format"
            ... )
        """
        PromptFormatter.FORMATS[name] = {
            "template": template,
            "stop_sequences": stop_sequences,
            "description": description,
            "system_message": system_message
        }
        print(f"✅ Added custom format: {name}")


# Example usage and testing
if __name__ == "__main__":
    # Test detection
    test_paths = [
        "tinyllama-1.1b.gguf",
        "llama-2-7b.Q4_K_M.gguf",
        "llama-3-8b.gguf",
        "mistral-7b-instruct.gguf",
        "qwen/qwen3-32b",
    ]
    
    for path in test_paths:
        family = PromptFormatter.detect_model_family(path)
        formatted = PromptFormatter.format_prompt("What is AI?", path)
        stops = PromptFormatter.get_stop_sequences(path)
        sys_msg = PromptFormatter.get_system_message(path)
        print(f"\n{path}:")
        print(f"  Family: {family}")
        print(f"  Stops: {stops}")
        print(f"  System: {sys_msg}")
        print(f"  Formatted: {formatted[:100]}...")