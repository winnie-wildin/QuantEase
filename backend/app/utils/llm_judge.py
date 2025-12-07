#app/utils/llm_judge.py
"""
LLM-as-Judge evaluation using gpt-oss-120b via Groq API.
Used for optional qualitative assessment (10% sampling by default).
"""
from typing import List, Dict, Tuple
import random
from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq

class LLMJudge:
    """
    LLM Judge for qualitative evaluation of Text Generation and RAG tasks.
    Uses gpt-oss-120b from Groq for cost-effective judging.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the judge with Groq API.
        
        Args:
            api_key: Groq API key (or from environment)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required for LLM Judge")
        
        self.client = Groq(api_key=self.api_key)
        self.judge_model = "openai/gpt-oss-120b"
    
    def sample_data(
        self,
        data: List[Dict],
        sample_percentage: float = 10.0,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[int]]:
        """
        Randomly sample data for judge evaluation.
        
        Args:
            data: List of data points to sample from
            sample_percentage: Percentage to sample (default 10%)
            random_seed: Random seed for reproducibility
        
        Returns:
            (sampled_data, sampled_indices)
        """
        random.seed(random_seed)
        
        total_samples = len(data)
        num_samples = max(1, int(total_samples * sample_percentage / 100))
        
        # Random sampling
        sampled_indices = sorted(random.sample(range(total_samples), num_samples))
        sampled_data = [data[i] for i in sampled_indices]
        
        print(f"   🎲 Sampled {num_samples} of {total_samples} ({sample_percentage}%)")
        
        return sampled_data, sampled_indices
    def _parse_judge_response(self, response_text: str) -> Dict:
        """
        Safely parse LLM judge response, handling markdown and malformed JSON.
        
        Args:
            response_text: Raw response from LLM
        
        Returns:
            Parsed JSON dict or error dict
        """
        import json
        import re
        
        try:
            # Remove markdown code fences
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                # Extract content between ```json and ```
                match = re.search(r'```(?:json)?\s*\n(.*?)\n```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
                else:
                    # Just remove the backticks
                    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            
            # Try to parse
            result = json.loads(cleaned)
            return result
            
        except json.JSONDecodeError as e:
            # Try to extract JSON object with regex
            try:
                # Find the first complete JSON object in the text
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                    return result
            except:
                pass
            
            print(f"   ⚠️ JSON parse error: {e}")
            print(f"   📝 Raw response: {response_text[:200]}...")
            return {"error": f"JSON parse error: {str(e)}"}
        
        except Exception as e:
            print(f"   ⚠️ Unexpected error: {e}")
            return {"error": str(e)}
    
    def judge_text_generation(
        self,
        input_text: str,
        reference_output: str,
        candidate_output: str,
        task_context: str = "general text generation"
    ) -> Dict:
        """
        Judge text generation quality using LLM.
        
        Args:
            input_text: Original input/prompt
            reference_output: Ground truth or baseline output
            candidate_output: Output from quantized model
            task_context: Description of the task (e.g., "translation", "summarization")
        
        Returns:
            Dict with judge scores
        """
        prompt = f"""You are an expert evaluator for {task_context} tasks.

INPUT:
{input_text}

REFERENCE OUTPUT:
{reference_output}

CANDIDATE OUTPUT:
{candidate_output}

Evaluate the CANDIDATE OUTPUT compared to the REFERENCE OUTPUT on these criteria:

1. ACCURACY (1-5): Does it convey the same information?
2. FLUENCY (1-5): Is it grammatically correct and natural?
3. COHERENCE (1-5): Is it logically structured and clear?

Respond ONLY with a JSON object:
{{
  "accuracy": <score 1-5>,
  "fluency": <score 1-5>,
  "coherence": <score 1-5>,
  "reasoning": "<brief explanation>"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
                reasoning_effort="medium",
                response_format={"type": "json_object"}  # ✅ Force JSON mode
            )
            
            import json
            response_text = response.choices[0].message.content
            return self._parse_judge_response(response_text)  # ✅ Use robust parser
            
        except Exception as e:
            print(f"   ⚠️ Judge error: {e}")
            return {"error": str(e)}
    
    def judge_rag_factuality(
        self,
        question: str,
        context: str,
        reference_answer: str,
        candidate_answer: str
    ) -> Dict:
        """
        Judge RAG/QA answer for hallucinations and factual correctness.
        
        Args:
            question: The question asked
            context: Retrieved context
            reference_answer: Ground truth answer
            candidate_answer: Model's answer
        
        Returns:
            Dict with hallucination detection and correctness score
        """
        prompt = f"""You are an expert fact-checker for question-answering systems.

QUESTION:
{question}

CONTEXT:
{context}

REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{candidate_answer}

Evaluate the CANDIDATE ANSWER:

1. HALLUCINATION: Does it contain information NOT present in the context? (yes/no)
2. FACTUAL CORRECTNESS: Is the answer factually correct given the context? (1-5)
3. COMPLETENESS: Does it answer the full question? (1-5)

Respond ONLY with a JSON object:
{{
  "has_hallucination": <true/false>,
  "factual_correctness": <score 1-5>,
  "completeness": <score 1-5>,
  "reasoning": "<brief explanation>"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
                reasoning_effort="medium",
                response_format={"type": "json_object"}  # ✅ Force JSON mode
            )
            
            import json
            response_text = response.choices[0].message.content
            return self._parse_judge_response(response_text)  # ✅ Use robust parser
            
        except Exception as e:
            print(f"   ⚠️ Judge error: {e}")
            return {"error": str(e)}
    
    def aggregate_judge_scores(self, judge_results: List[Dict], task_type: str) -> Dict:
        """
        Aggregate judge scores across all sampled data points.
        
        Args:
            judge_results: List of individual judge results
            task_type: "text_gen" or "rag"
        
        Returns:
            Aggregated statistics
        """
        # Filter out errors
        valid_results = [r for r in judge_results if "error" not in r]
        
        if not valid_results:
            return {"error": "All judge evaluations failed"}
        
        if task_type == "text_gen":
            return {
                "avg_accuracy": np.mean([r["accuracy"] for r in valid_results]),
                "avg_fluency": np.mean([r["fluency"] for r in valid_results]),
                "avg_coherence": np.mean([r["coherence"] for r in valid_results]),
                "num_evaluated": len(valid_results),
                "num_failed": len(judge_results) - len(valid_results)
            }
        
        elif task_type == "rag":
            hallucination_count = sum(1 for r in valid_results if r.get("has_hallucination", False))
            return {
                "hallucination_rate": (hallucination_count / len(valid_results)) * 100,
                "avg_factual_correctness": np.mean([r["factual_correctness"] for r in valid_results]),
                "avg_completeness": np.mean([r["completeness"] for r in valid_results]),
                "num_evaluated": len(valid_results),
                "num_failed": len(judge_results) - len(valid_results)
            }
        
        return {}


# Add numpy import at the top
import numpy as np