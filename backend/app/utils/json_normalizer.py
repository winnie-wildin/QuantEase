#app/utils/json_normalizer.py
"""
Normalize user-uploaded JSON to standard format based on task type.
Converts different key names (e.g., 'output' → 'label' for classification).
"""
from typing import List, Dict, Tuple
import json


class JSONNormalizer:
    """
    Normalize uploaded JSON data to task-specific standard formats.
    """
    
    # Standard keys for each task type
    TASK_SCHEMAS = {
        "text_generation": {
            "input_key": "input",
            "output_key": "output"
        },
        "classification": {
            "input_key": "input",
            "output_key": "label"
        },
        "rag": {
            "input_key": "input",  # Will contain question
            "context_key": "context",
            "output_key": "output"  # Will contain answer
        }
    }
    
    @staticmethod
    def detect_keys(sample: Dict) -> Tuple[str, str, bool]:
        """
        Detect input and output keys from a sample JSON object.
        
        Args:
            sample: Single JSON object from user upload
        
        Returns:
            (input_key, output_key, has_output)
        """
        keys = list(sample.keys())
        
        # Common input key variations
        input_variations = ['input', 'input_text', 'text', 'prompt', 'question']
        # Common output key variations
        output_variations = ["output", "label", "answer", "response", "target", "ground_truth", "expected_output", 'ground_truth_output']
        
        input_key = None
        output_key = None
        
        # Find input key
        for key in keys:
            if key.lower() in input_variations:
                input_key = key
                break
        
        # Find output key
        for key in keys:
            if key.lower() in output_variations and key != input_key:
                output_key = key
                break
        
        has_output = output_key is not None
        
        return input_key, output_key, has_output
    
    @staticmethod
    def normalize_text_generation(data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Normalize JSON for text generation task.
        
        Expected output format:
        [
          {"input": "...", "output": "..."} or {"input": "..."}
        ]
        
        Args:
            data: User-uploaded JSON
        
        Returns:
            (normalized_data, metadata)
        """
        if not data:
            raise ValueError("Empty dataset")
        
        # Detect keys from first sample
        input_key, output_key, has_output = JSONNormalizer.detect_keys(data[0])
        
        if not input_key:
            raise ValueError("Could not detect input key. Use 'input', 'text', or 'prompt'")
        
        # Normalize all samples
        normalized = []
        for item in data:
            normalized_item = {"input": item[input_key]}
            
            if has_output and output_key in item:
                normalized_item["output"] = item[output_key]
            
            normalized.append(normalized_item)
        
        metadata = {
            "original_input_key": input_key,
            "original_output_key": output_key if has_output else None,
            "has_ground_truth": has_output,
            "num_samples": len(normalized)
        }
        
        return normalized, metadata
    
    @staticmethod
    def normalize_classification(data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Normalize JSON for classification task.
        
        Expected output format:
        [
          {"input": "...", "label": "..."} or {"input": "..."}
        ]
        
        Args:
            data: User-uploaded JSON
        
        Returns:
            (normalized_data, metadata)
        """
        if not data:
            raise ValueError("Empty dataset")
        
        # Detect keys
        input_key, output_key, has_output = JSONNormalizer.detect_keys(data[0])
        
        if not input_key:
            raise ValueError("Could not detect input key")
        
        # Normalize all samples
        normalized = []
        all_labels = []
        
        for item in data:
            normalized_item = {"input": item[input_key]}
            
            if has_output and output_key in item:
                label = str(item[output_key])  # Ensure label is string
                normalized_item["label"] = label
                all_labels.append(label)
            
            normalized.append(normalized_item)
        
        # Calculate class statistics if labels exist
        class_stats = None
        if all_labels:
            from collections import Counter
            label_counts = Counter(all_labels)
            class_stats = {
                "num_classes": len(label_counts),
                "class_distribution": dict(label_counts),
                "most_common": label_counts.most_common(1)[0] if label_counts else None,
                "least_common": label_counts.most_common()[-1] if label_counts else None
            }
        
        metadata = {
            "original_input_key": input_key,
            "original_output_key": output_key if has_output else None,
            "has_ground_truth": has_output,
            "num_samples": len(normalized),
            "class_statistics": class_stats
        }
        
        return normalized, metadata
    
    @staticmethod
    def normalize_rag(data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Normalize JSON for RAG task.
        
        Expected output format:
        [
          {
            "input": "question",
            "context": "retrieved context",
            "output": "answer"
          }
        ]
        
        Args:
            data: User-uploaded JSON
        
        Returns:
            (normalized_data, metadata)
        """
        if not data:
            raise ValueError("Empty dataset")
        
        sample = data[0]
        keys = list(sample.keys())
        
        # Detect input (question) - NOW INCLUDES input_text
        input_key = None
        for key in ["input", "input_text", "question", "query", "prompt", "text"]:
            if key in keys:
                input_key = key
                break

        # Detect context
        context_key = None
        for key in ["context", "retrieved_context", "passage", "document"]:
            if key in keys:
                context_key = key
                break

        # Detect output (answer) - NOW INCLUDES ground_truth_output
        output_key = None
        has_output = False
        for key in ["output", "answer", "response", "ground_truth", "ground_truth_output"]:
            if key in keys:
                output_key = key
                has_output = True
                break
        if not input_key:
            raise ValueError("Could not detect input/question key")
        
        if not context_key:
            raise ValueError("RAG tasks require 'context' field")
        
        # Normalize all samples
        normalized = []
        for item in data:
            normalized_item = {
                "input": item[input_key],
                "context": item[context_key]
            }
            
            if has_output and output_key in item:
                normalized_item["output"] = item[output_key]
            
            normalized.append(normalized_item)
        
        metadata = {
            "original_input_key": input_key,
            "original_context_key": context_key,
            "original_output_key": output_key if has_output else None,
            "has_ground_truth": has_output,
            "num_samples": len(normalized)
        }
        
        return normalized, metadata
    
    @staticmethod
    def normalize_by_task(data: List[Dict], task_type: str) -> Tuple[List[Dict], Dict]:
        """
        Normalize JSON based on task type.
        
        Args:
            data: User-uploaded JSON
            task_type: One of ["text_generation", "classification", "rag"]
        
        Returns:
            (normalized_data, metadata)
        """
        if task_type == "text_generation":
            return JSONNormalizer.normalize_text_generation(data)
        elif task_type == "classification":
            return JSONNormalizer.normalize_classification(data)
        elif task_type == "rag":
            return JSONNormalizer.normalize_rag(data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")