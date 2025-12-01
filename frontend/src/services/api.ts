// frontend/src/services/api.ts
import type { Experiment, ModelVariant, ComparativeMetrics } from '../types';

const API_BASE_URL = 'http://localhost:8000';

interface UploadSamplesResponse {
  message: string;
  sample_count: number;
  has_ground_truth: boolean;
  task_type: string;
  metadata: any;
}

export const api = {
  // Experiments
  getExperiments: async (): Promise<Experiment[]> => {
    const response = await fetch(`${API_BASE_URL}/experiments/`);
    if (!response.ok) throw new Error('Failed to fetch experiments');
    return response.json();
  },

  deleteExperiment: async (experimentId: number) => {
    const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete experiment');
    return response.json();
  },

  getExperiment: async (id: number): Promise<Experiment> => {
    const response = await fetch(`${API_BASE_URL}/experiments/${id}`);
    if (!response.ok) throw new Error('Experiment not found');
    return response.json();
  },

  createExperiment: async (data: {
    name: string;
    baseline_model_id: number;
    has_ground_truth: boolean;
  }): Promise<Experiment> => {
    const response = await fetch(`${API_BASE_URL}/experiments/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to create experiment');
    return response.json();
  },

  // Variants
  getVariants: async (experimentId: number): Promise<ModelVariant[]> => {
    const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}/variants`);
    if (!response.ok) throw new Error('Failed to fetch variants');
    return response.json();
  },

  createVariant: async (experimentId: number, data: {
    variant_type: string;
    model_name: string;
    quantization_level?: string;
    model_path?: string;
    inference_provider: string;
  }): Promise<ModelVariant> => {
    const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}/variants`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to create variant');
    return response.json();
  },

  // Metrics
  getExperimentMetrics: async (experimentId: number): Promise<ComparativeMetrics[]> => {
    const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}/metrics`);
    if (!response.ok) throw new Error('Failed to fetch metrics');
    return response.json();
  },

  // Samples - UPDATED with task_type
  uploadSamples: async (
    experimentId: number, 
    samples: any[], 
    taskType: string
  ): Promise<UploadSamplesResponse> => {
    const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}/samples`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        samples,
        task_type: taskType  // ← Added task_type
      }),
    });
    if (!response.ok) throw new Error('Failed to upload samples');
    return response.json();
  },

  // Generation
  triggerGeneration: async (experimentId: number): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}/generate`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to trigger generation');
  },

  triggerEvaluation: async (
    experimentId: number,
    enableLLMJudge: boolean = false
  ): Promise<void> => {
    const response = await fetch(
      `${API_BASE_URL}/experiments/${experimentId}/evaluate?enable_llm_judge=${enableLLMJudge}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }
    );
    if (!response.ok) throw new Error('Failed to trigger evaluation');
  },
};