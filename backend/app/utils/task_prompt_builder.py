# app/utils/task_prompt_builder.py
"""
Task Prompt Builder - Task-specific prompt construction
Builds prompts tailored to different task types (classification, text generation, RAG)
"""
from typing import List, Optional, Dict


class TaskPromptBuilder:
    """
    Build task-aware prompts for different evaluation tasks.
    
    This handles CONTENT/INSTRUCTIONS (what to do),
    while PromptFormatter handles SYNTAX (how to format for specific models).
    """
    
    @staticmethod
    def build(
        task_type: str,
        input_text: str,
        labels: Optional[List[str]] = None,
        context: Optional[str] = None,
        system_message: Optional[str] = None
    ) -> str:
        """
        Build a task-specific prompt.
        
        Args:
            task_type: One of 'text_generation', 'classification', 'rag'
            input_text: The user's input/question
            labels: List of valid labels (required for classification)
            context: Document context (required for RAG)
            system_message: Optional model-specific system message
            
        Returns:
            Task-formatted prompt string
            
        Raises:
            ValueError: If required parameters are missing for task type
        """
        if task_type == "classification":
            return TaskPromptBuilder._build_classification_prompt(input_text, labels)
        elif task_type == "text_generation":
            return TaskPromptBuilder._build_text_generation_prompt(input_text)
        elif task_type == "rag":
            return TaskPromptBuilder._build_rag_prompt(input_text, context)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @staticmethod
    def _build_classification_prompt(input_text: str, labels: Optional[List[str]] = None) -> str:
        """
        Build classification prompt that forces single-label output.
        
        Args:
            input_text: Text to classify
            labels: List of valid category labels
            
        Returns:
            Classification prompt
        """
        if not labels or len(labels) == 0:
            raise ValueError("Classification task requires 'labels' parameter")
        
        # Format labels nicely
        labels_str = ", ".join(labels)
        
        prompt = f"""Classify the following text into exactly ONE of these categories: [{labels_str}]

CRITICAL INSTRUCTIONS:
- Output ONLY the category label
- Do NOT include explanations, reasoning, or additional text
- Do NOT use punctuation or formatting
- Your entire response must be a single word from the list above

Text to classify: {input_text}

Category:"""
        
        return prompt.strip()
    
    @staticmethod
    def _build_text_generation_prompt(input_text: str) -> str:
        """
        Build text generation prompt (straightforward).
        
        Args:
            input_text: The user's prompt/question
            
        Returns:
            Text generation prompt
        """
        # For text generation, we just use the input as-is
        # No special instructions needed - let the model be creative
        return input_text.strip()
    
    @staticmethod
    def _build_rag_prompt(input_text: str, context: Optional[str] = None) -> str:
        """
        Build RAG (Retrieval-Augmented Generation) prompt.
        
        Args:
            input_text: The question to answer
            context: Retrieved document/passage
            
        Returns:
            RAG prompt with context
        """
        if not context:
            raise ValueError("RAG task requires 'context' parameter")
        
        prompt = f"""Answer the following question based ONLY on the information provided in the context below. Do not use external knowledge.

Context:
{context}

Question: {input_text}

Answer:"""
        
        return prompt.strip()
    
    @staticmethod
    def extract_labels_from_samples(samples: List[Dict]) -> List[str]:
        """
        Extract unique labels from classification samples.
        
        Args:
            samples: List of dicts with 'ground_truth_output' or 'label' field
            
        Returns:
            Sorted list of unique labels
            
        Example:
            >>> samples = [
            ...     {"input_text": "Great!", "ground_truth_output": "positive"},
            ...     {"input_text": "Bad", "ground_truth_output": "negative"},
            ...     {"input_text": "OK", "ground_truth_output": "neutral"},
            ...     {"input_text": "Love it!", "ground_truth_output": "positive"}
            ... ]
            >>> TaskPromptBuilder.extract_labels_from_samples(samples)
            ['negative', 'neutral', 'positive']
        """
        labels = set()
        
        for sample in samples:
            # Try different field names
            label = sample.get('ground_truth_output') or sample.get('label') or sample.get('output')
            
            if label:
                # Strip whitespace and convert to lowercase for consistency
                label = str(label).strip()
                if label:  # Only add non-empty labels
                    labels.add(label)
        
        if not labels:
            raise ValueError("No labels found in samples. Classification requires ground truth labels.")
        
        # Return sorted list for consistent ordering
        return sorted(list(labels))
    
    @staticmethod
    def validate_task_data(task_type: str, samples: List[Dict]) -> Dict:
        """
        Validate that samples have required fields for task type.
        
        Args:
            task_type: The task type
            samples: List of sample dicts
            
        Returns:
            Dict with validation results and extracted metadata
            
        Raises:
            ValueError: If validation fails
        """
        if task_type == "classification":
            # Must have labels
            labels = TaskPromptBuilder.extract_labels_from_samples(samples)
            
            return {
                "valid": True,
                "labels": labels,
                "num_classes": len(labels),
                "message": f"Found {len(labels)} classes: {labels}"
            }
        
        elif task_type == "rag":
            # Must have context field
            missing_context = []
            for i, sample in enumerate(samples):
                if not sample.get('context'):
                    missing_context.append(i)
            
            if missing_context:
                raise ValueError(
                    f"RAG task requires 'context' field in all samples. "
                    f"Missing in samples: {missing_context[:5]}..."
                )
            
            return {
                "valid": True,
                "message": "All samples have context field"
            }
        
        elif task_type == "text_generation":
            # No special requirements
            return {
                "valid": True,
                "message": "Text generation task validated"
            }
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("TASK PROMPT BUILDER - EXAMPLES")
    print("="*60)
    
    # Example 1: Classification
    print("\n1. CLASSIFICATION TASK:")
    print("-" * 60)
    classification_prompt = TaskPromptBuilder.build(
        task_type="classification",
        input_text="This product is absolutely amazing! Best purchase ever!",
        labels=["positive", "negative", "neutral"]
    )
    print(classification_prompt)
    
    # Example 2: Text Generation
    print("\n2. TEXT GENERATION TASK:")
    print("-" * 60)
    text_gen_prompt = TaskPromptBuilder.build(
        task_type="text_generation",
        input_text="Explain quantum entanglement in simple terms."
    )
    print(text_gen_prompt)
    
    # Example 3: RAG
    print("\n3. RAG TASK:")
    print("-" * 60)
    rag_prompt = TaskPromptBuilder.build(
        task_type="rag",
        input_text="When was the Eiffel Tower built?",
        context="The Eiffel Tower was constructed between 1887 and 1889 for the 1889 World's Fair in Paris."
    )
    print(rag_prompt)
    
    # Example 4: Label Extraction
    print("\n4. LABEL EXTRACTION:")
    print("-" * 60)
    samples = [
        {"input_text": "Great product!", "ground_truth_output": "positive"},
        {"input_text": "Terrible quality", "ground_truth_output": "negative"},
        {"input_text": "It's okay", "ground_truth_output": "neutral"},
        {"input_text": "Love it!", "ground_truth_output": "positive"}
    ]
    labels = TaskPromptBuilder.extract_labels_from_samples(samples)
    print(f"Extracted labels: {labels}")
    
    # Example 5: Validation
    print("\n5. TASK VALIDATION:")
    print("-" * 60)
    validation = TaskPromptBuilder.validate_task_data("classification", samples)
    print(f"Validation result: {validation}")
    
    print("\n" + "="*60)