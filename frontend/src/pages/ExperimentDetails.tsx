import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, RefreshCw, Trash2, Loader2, CheckCircle2, FileText } from 'lucide-react';import toast from 'react-hot-toast';
import { api } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';

interface Experiment {
  id: number;
  name: string;
  baseline_model_id: number;
  has_ground_truth: boolean;
  sample_count: number;
  status: string;
  progress: number;
  created_at: string;
  updated_at: string;
}

interface Variant {
  id: number;
  experiment_id: number;
  variant_type: string;
  model_name: string;
  quantization_level: string | null;
  inference_provider: string;
  status: string;
  progress: number;
  display_name: string;
}

interface Metric {
  id: number;
  variant_id: number;
  model_size_mb: number | null;
  avg_latency_ms: number | null;
  avg_token_count: number | null;
  avg_tokens_per_second: number | null;
  bertscore_f1_vs_gt: number | null;
  bertscore_f1_vs_baseline: number | null;
  cosine_similarity_vs_gt: number | null;
  cosine_similarity_vs_baseline: number | null;
  output_divergence_score: number | null;
  samples_evaluated: number;
  evaluation_status: string;
}

export const ExperimentDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const experimentId = parseInt(id || '0');

  const [evaluating, setEvaluating] = useState(false);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(false);

  // Fetch experiment data
  const { 
    data: experiment, 
    isLoading, 
    refetch: refetchExperiment 
  } = useQuery<Experiment>({
    queryKey: ['experiment', experimentId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:8000/experiments/${experimentId}`);
      if (!response.ok) throw new Error('Failed to fetch experiment');
      return response.json();
    },
    refetchInterval: autoRefreshEnabled ? 3000 : false, // Auto-refresh every 3s when enabled
  });

  // Fetch variants
  const { 
    data: variants = [], 
    isLoading: isLoadingVariants,
    refetch: refetchVariants 
  } = useQuery<Variant[]>({
    queryKey: ['variants', experimentId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:8000/experiments/${experimentId}/variants`);
      if (!response.ok) throw new Error('Failed to fetch variants');
      return response.json();
    },
    refetchInterval: autoRefreshEnabled ? 3000 : false,
  });

  // Fetch metrics
  const { 
    data: metrics = [], 
    refetch: refetchMetrics 
  } = useQuery<Metric[]>({
    queryKey: ['metrics', experimentId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:8000/experiments/${experimentId}/metrics`);
      if (!response.ok) throw new Error('Failed to fetch metrics');
      return response.json();
    },
    refetchInterval: autoRefreshEnabled ? 3000 : false,
  });

  // Enable auto-refresh when generating or evaluating
  useEffect(() => {
    const isGenerating = experiment?.status === 'generating' || 
                         variants.some(v => v.status === 'generating');
    const isEvaluating = evaluating;
    
    setAutoRefreshEnabled(isGenerating || isEvaluating);
  }, [experiment, variants, evaluating]);

  // Check if generation is complete but evaluation hasn't run
  const needsEvaluation = experiment?.status === 'completed' && metrics.length === 0;

  // Handle evaluation
  const handleEvaluation = async () => {
    setEvaluating(true);

    // Show immediate toast
    const toastId = toast.loading('🧮 Evaluation started! Calculating BERTScore...', {
      duration: Infinity,
    });

    try {
      await api.triggerEvaluation(experimentId);

      // Poll for completion (max 2 minutes)
      let attempts = 0;
      const maxAttempts = 40; // 40 * 3s = 2 minutes

      const pollInterval = setInterval(async () => {
        attempts++;
        
        const newMetrics = await refetchMetrics();
        
        if (newMetrics.data && newMetrics.data.length > 0) {
          // Evaluation complete!
          clearInterval(pollInterval);
          setEvaluating(false);
          toast.success('✅ Evaluation complete!', { id: toastId });
          refetchVariants();
          refetchExperiment();
        } else if (attempts >= maxAttempts) {
          // Timeout
          clearInterval(pollInterval);
          setEvaluating(false);
          toast.error('⚠️ Evaluation taking longer than expected. Refresh to check status.', { 
            id: toastId,
            duration: 5000,
          });
        }
      }, 3000); // Check every 3 seconds

    } catch (error) {
      console.error('Evaluation error:', error);
      toast.error('❌ Evaluation failed: ' + error, { id: toastId });
      setEvaluating(false);
    }
  };

  // Handle delete
  const handleDelete = async () => {
    if (!window.confirm('⚠️ Delete this experiment? This cannot be undone.')) {
      return;
    }

    const toastId = toast.loading('Deleting experiment...');
    try {
      await api.deleteExperiment(experimentId);
      toast.success('✅ Experiment deleted', { id: toastId });
      navigate('/');
    } catch (error) {
      toast.error('❌ Failed to delete: ' + error, { id: toastId });
    }
  };

  // Handle manual refresh
  const handleRefresh = async () => {
    const toastId = toast.loading('Refreshing...');
    try {
      await Promise.all([
        refetchExperiment(),
        refetchVariants(),
        refetchMetrics(),
      ]);
      toast.success('✅ Refreshed', { id: toastId, duration: 1500 });
    } catch (error) {
      toast.error('❌ Failed to refresh', { id: toastId });
    }
  };

  // Get baseline and quantized variants
  const baselineVariant = variants.find(v => v.variant_type === 'baseline');
  const quantizedVariants = variants.filter(v => v.variant_type === 'quantized');

  // Get metrics for each variant
  const getMetricsForVariant = (variantId: number) => {
    return metrics.find(m => m.variant_id === variantId);
  };

  // Loading state
  if (isLoading || isLoadingVariants) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading experiment...</p>
        </div>
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">Experiment not found</p>
        <Link to="/" className="text-primary-600 hover:underline mt-4 inline-block">
          ← Back to experiments
        </Link>
      </div>
    );
  }

  // Check if still generating
  const isGenerating = experiment.status === 'generating' || 
                       variants.some(v => v.status === 'generating');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link to="/" className="text-gray-600 hover:text-gray-900 transition-colors">
            <ArrowLeft className="w-6 h-6" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{experiment.name}</h1>
            <p className="text-gray-600">{experiment.sample_count} samples</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* Status Indicators */}
          <div className="flex flex-col items-end space-y-2">
            {/* Generation Status */}
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-500 uppercase tracking-wide font-medium">
                Generation
              </span>
              <StatusBadge status={experiment.status} />
            </div>

            {/* Evaluation Status */}
            {evaluating && (
              <div className="flex items-center space-x-2">
                <span className="text-xs text-blue-500 uppercase tracking-wide font-medium">
                  Evaluation
                </span>
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800 animate-pulse">
                  <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                  Running...
                </span>
              </div>
            )}

            {!evaluating && metrics.length > 0 && (
              <div className="flex items-center space-x-2">
                <span className="text-xs text-green-500 uppercase tracking-wide font-medium">
                  Evaluation
                </span>
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                  <CheckCircle2 className="w-4 h-4 mr-1" />
                  Complete
                </span>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <button
            onClick={handleRefresh}
            disabled={autoRefreshEnabled}
            className="btn-secondary flex items-center space-x-2"
            title="Refresh data"
          >
            <RefreshCw className={`w-4 h-4 ${autoRefreshEnabled ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleDelete}
            className="btn-secondary text-red-600 hover:bg-red-50 flex items-center space-x-2"
          >
            <Trash2 className="w-4 h-4" />
            <span>Delete</span>
          </button>
        </div>
      </div>

      {/* Auto-refresh indicator */}
      {autoRefreshEnabled && (
        <div className="card bg-blue-50 border-blue-200">
          <div className="flex items-center space-x-2 text-blue-700">
            <Loader2 className="w-4 h-4 animate-spin" />
            <p className="text-sm font-medium">
              Auto-refreshing every 3 seconds...
            </p>
          </div>
        </div>
      )}

      {/* Generation in progress */}
      {isGenerating && (
        <div className="card bg-yellow-50 border-yellow-200">
          <div className="flex items-center space-x-3">
            <Loader2 className="w-5 h-5 animate-spin text-yellow-600" />
            <div>
              <p className="text-yellow-800 font-medium">🚀 Generation in progress...</p>
              <p className="text-sm text-yellow-700 mt-1">
                This may take 1-3 minutes depending on model size. Page auto-refreshing.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Evaluation needed */}
      {needsEvaluation && !isGenerating && (
        <div className="card bg-yellow-50 border-yellow-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-yellow-800 font-medium">
                ⚠️ Generation complete. Run evaluation to see metrics.
              </p>
              <p className="text-sm text-yellow-700 mt-1">
                Evaluation takes 60-90 seconds (calculating BERTScore with AI models)
              </p>
            </div>
            <button
              onClick={handleEvaluation}
              disabled={evaluating}
              className="btn-primary whitespace-nowrap flex items-center space-x-2"
            >
              {evaluating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Running...</span>
                </>
              ) : (
                <>
                  <span>🧮 Run Evaluation</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* View Sample Outputs Button */}
      {metrics.length > 0 && (
        <Link
          to={`/experiment/${experimentId}/samples`}
          className="btn-primary inline-flex items-center space-x-2"
        >
          <FileText className="w-4 h-4" />
          <span>View Sample Outputs</span>
        </Link>
      )}

      {/* Variant Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Baseline Card */}
        {baselineVariant && (
          <div className="card">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Baseline (Groq)</h3>
            <p className="text-gray-600 mb-4">{baselineVariant.model_name}</p>

            {baselineVariant.status === 'generating' && (
              <div className="flex items-center space-x-2 text-blue-600 mb-4">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm">Generating...</span>
              </div>
            )}

            {getMetricsForVariant(baselineVariant.id) ? (
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 flex items-center">
                    <span className="mr-2">⚡</span> Speed
                  </span>
                  <span className="font-semibold text-gray-900">
                    {getMetricsForVariant(baselineVariant.id)?.avg_tokens_per_second?.toFixed(1)} tok/s
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 flex items-center">
                    <span className="mr-2">⏱️</span> Latency
                  </span>
                  <span className="font-semibold text-gray-900">
                    {getMetricsForVariant(baselineVariant.id)?.avg_latency_ms?.toFixed(0)}ms
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 flex items-center">
                    <span className="mr-2">📍</span> Role
                  </span>
                  <span className="inline-flex items-center px-2 py-1 rounded text-sm font-medium bg-pink-100 text-pink-700">
                    📌 Reference
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-gray-500 text-sm">Waiting for metrics...</p>
            )}
          </div>
        )}

        {/* Quantized Cards */}
        {quantizedVariants.map((variant) => {
          const variantMetrics = getMetricsForVariant(variant.id);

          return (
            <div key={variant.id} className="card">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Quantized (GGUF)</h3>
              <p className="text-gray-600 mb-4">
                {variant.model_name} • {variant.quantization_level}
              </p>

              {variant.status === 'generating' && (
                <div className="flex items-center space-x-2 text-blue-600 mb-4">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Generating...</span>
                </div>
              )}

              {variantMetrics ? (
                <div className="space-y-3">
                  {/* Speed */}
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 flex items-center">
                      <span className="mr-2">⚡</span> Speed
                    </span>
                    <span className="font-semibold text-gray-900">
                      {variantMetrics.avg_tokens_per_second?.toFixed(1)} tok/s
                    </span>
                  </div>

                  {/* Latency */}
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 flex items-center">
                      <span className="mr-2">⏱️</span> Latency
                    </span>
                    <span className="font-semibold text-gray-900">
                      {variantMetrics.avg_latency_ms?.toFixed(0)}ms
                    </span>
                  </div>

                  {/* Model Size */}
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 flex items-center">
                      <span className="mr-2">💾</span> Model Size
                    </span>
                    <span className="font-semibold text-gray-900">
                      {variantMetrics.model_size_mb?.toFixed(0)}MB
                    </span>
                  </div>

                  {/* Quality (BERTScore) */}
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600 flex items-center">
                      <span className="mr-2">🎯</span> Quality (BERTScore)
                    </span>
                    <span className="font-semibold text-gray-900">
                      {((variantMetrics.bertscore_f1_vs_baseline || variantMetrics.bertscore_f1_vs_gt || 0) * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="border-t border-gray-200 pt-3 mt-3">
                    <p className="text-xs text-gray-500 font-medium mb-2">Additional Metrics</p>
                    
                    {/* Cosine Similarity */}
                    {(variantMetrics.cosine_similarity_vs_baseline || variantMetrics.cosine_similarity_vs_gt) && (
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">Similarity</span>
                        <span className="text-gray-700">
                          {((variantMetrics.cosine_similarity_vs_baseline || variantMetrics.cosine_similarity_vs_gt || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}

                    {/* Divergence */}
                    {variantMetrics.output_divergence_score !== null && (
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">Divergence</span>
                        <span className="text-gray-700">
                          {((variantMetrics.output_divergence_score || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}

                    {/* Avg Token Count */}
                    {variantMetrics.avg_token_count && (
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">Avg Tokens</span>
                        <span className="text-gray-700">
                          {variantMetrics.avg_token_count.toFixed(0)}
                        </span>
                      </div>
                    )}

                    {/* BERTScore Details (optional - guard with any cast since interface may not include all fields) */}
                    {(variantMetrics as any).bertscore_precision_vs_baseline && (
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">BERT Precision</span>
                        <span className="text-gray-700">
                          {((variantMetrics as any).bertscore_precision_vs_baseline * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}

                    {(variantMetrics as any).bertscore_recall_vs_baseline && (
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-500">BERT Recall</span>
                        <span className="text-gray-700">
                          {((variantMetrics as any).bertscore_recall_vs_baseline * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <p className="text-gray-500 text-sm">Waiting for metrics...</p>
              )}
            </div>
          );
        })}
      </div>

      {/* Comparison Charts - Only show if metrics available */}
      {metrics.length > 0 && baselineVariant && quantizedVariants.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Speed Comparison */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Speed Comparison</h3>
            <div className="space-y-4">
              {/* Baseline */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">Baseline - {baselineVariant.model_name}</span>
                  <span className="font-medium">
                    {getMetricsForVariant(baselineVariant.id)?.avg_tokens_per_second?.toFixed(1)} tok/s
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-blue-600 h-3 rounded-full"
                    style={{
                      width: `${Math.min(100, (getMetricsForVariant(baselineVariant.id)?.avg_tokens_per_second || 0) / 4)}%`,
                    }}
                  />
                </div>
              </div>

              {/* Quantized */}
              {quantizedVariants.map((variant) => {
                const variantMetrics = getMetricsForVariant(variant.id);
                return (
                  <div key={variant.id}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">
                        {variant.model_name} • {variant.quantization_level}
                      </span>
                      <span className="font-medium">
                        {variantMetrics?.avg_tokens_per_second?.toFixed(1)} tok/s
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className="bg-green-600 h-3 rounded-full"
                        style={{
                          width: `${Math.min(100, (variantMetrics?.avg_tokens_per_second || 0) / 4)}%`,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Quality Comparison */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quality Comparison</h3>
            <div className="space-y-4">
              {quantizedVariants.map((variant) => {
                const variantMetrics = getMetricsForVariant(variant.id);
                const quality = (variantMetrics?.bertscore_f1_vs_baseline || 0) * 100;
                return (
                  <div key={variant.id}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">
                        {variant.model_name} • {variant.quantization_level}
                      </span>
                      <span className="font-medium">{quality.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${
                          quality >= 70 ? 'bg-green-600' : quality >= 50 ? 'bg-yellow-600' : 'bg-red-600'
                        }`}
                        style={{ width: `${quality}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Trade-offs Summary */}
      {metrics.length > 0 && baselineVariant && quantizedVariants.length > 0 && (
        <div className="card bg-gradient-to-r from-yellow-50 to-orange-50 border-yellow-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <span className="mr-2">📊</span> Trade-offs Summary
          </h3>
          
          {/* Show summary for each quantized variant */}
          {quantizedVariants.map((variant, index) => {
            const variantMetrics = getMetricsForVariant(variant.id);
            const baselineMetrics = getMetricsForVariant(baselineVariant.id);
            const baselineSpeed = baselineMetrics?.avg_tokens_per_second || 0;
            const quantSpeed = variantMetrics?.avg_tokens_per_second || 0;
            const speedRatio = baselineSpeed / quantSpeed;
            const score = variantMetrics?.bertscore_f1_vs_baseline || variantMetrics?.bertscore_f1_vs_gt || 0;
            const div = variantMetrics?.output_divergence_score || 0;

            return (
              <div key={variant.id} className={index > 0 ? 'mt-4 pt-4 border-t border-yellow-300' : ''}>
                <h4 className="font-semibold text-gray-800 mb-3">
                  {variant.model_name} • {variant.quantization_level}
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {/* Speed */}
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-1">Speed</p>
                    <p className="text-gray-900">
                      Baseline is {speedRatio.toFixed(1)}x faster
                    </p>
                  </div>

                  {/* Size */}
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-1">Size</p>
                    <p className="text-gray-900">
                      Quantized uses only{' '}
                      {variantMetrics?.model_size_mb?.toFixed(0)}MB locally
                    </p>
                  </div>

                  {/* Quality */}
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-1">Quality Retention</p>
                    <p className="text-gray-900">
                      {(score * 100).toFixed(1)}% of baseline quality
                    </p>
                  </div>

                  {/* Divergence */}
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-1">Output Divergence</p>
                    <p className="text-gray-900">
                      {(div * 100).toFixed(1)}% different from baseline
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};