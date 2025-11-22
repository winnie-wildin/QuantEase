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
  bertscore_f1_vs_gt?: number;
  bertscore_f1_vs_baseline?: number;
  cosine_similarity_vs_gt?: number;
  cosine_similarity_vs_baseline?: number;
  output_divergence_score?: number;
  samples_evaluated: number;
  evaluation_status: string;
}