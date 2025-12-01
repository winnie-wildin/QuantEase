// frontend/src/types/index.ts

export type TaskType = 'text_generation' | 'classification' | 'rag';

export interface Experiment {
  id: number;
  name: string;
  baseline_model_id: number;
  has_ground_truth: boolean;
  sample_count: number;
  status: 'created' | 'generating' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  updated_at?: string;
  
  // NEW: Task-aware fields
  task_type?: TaskType;
  normalization_metadata?: {
    has_ground_truth: boolean;
    original_input_key?: string;
    original_output_key?: string;
    class_statistics?: {
      num_classes: number;
      class_distribution: Record<string, number>;
      is_imbalanced: boolean;
    };
  };
  judge_enabled?: boolean;
  judge_sample_percentage?: number;
  task_display_name?: string;
}

export interface ModelVariant {
  id: number;
  experiment_id: number;
  variant_type: 'baseline' | 'quantized' | 'bonus';
  model_name: string;
  quantization_level?: string;
  inference_provider: string;
  status: 'pending' | 'generating' | 'completed' | 'failed';
  progress: number;
  display_name: string;
}

export interface ComparativeMetrics {
  id: number;
  variant_id: number;
  model_size_mb?: number;
  avg_latency_ms?: number;
  avg_token_count?: number;
  avg_tokens_per_second?: number;
  
  // OLD metrics (deprecated but kept for backward compatibility)
  bertscore_f1_vs_gt?: number;
  bertscore_f1_vs_baseline?: number;
  cosine_similarity_vs_gt?: number;
  cosine_similarity_vs_baseline?: number;
  output_divergence_score?: number;
  
  samples_evaluated: number;
  evaluation_status: string;
  
  // NEW: Task-aware evaluation results
  // NEW: Task-aware evaluation results
  evaluation_results?: {
    // Text Generation
    bertscore_f1?: number;
    bertscore_precision?: number;
    bertscore_recall?: number;
    length_ratio_mean?: number;
    length_ratio_std?: number;
    divergence_score?: number;
    reference_type?: 'ground_truth' | 'baseline';
    
    // Classification
    accuracy?: number;
    macro_f1?: number;
    weighted_f1?: number;
    per_class_f1?: Record<string, number>;
    confusion_matrix?: number[][];
    is_imbalanced?: boolean;
    
    // RAG
    answer_relevance?: number;
    
    // ✅ UPDATED: Complete LLM Judge fields
    llm_judge?: {
      // Text Generation metrics (1-5 scale)
      avg_accuracy?: number;
      avg_fluency?: number;
      avg_coherence?: number;
      
      // RAG metrics
      hallucination_rate?: number;  // Already a percentage (0-100)
      avg_factual_correctness?: number;
      avg_completeness?: number;
      
      // Common fields
      num_evaluated?: number;
      num_failed?: number;
      sample_percentage?: number;
      sampled_indices?: number[];
      error?: string;
      
      // Legacy/optional
      avg_relevance?: number;
    };
  };
}