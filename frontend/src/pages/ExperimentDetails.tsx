// frontend/src/pages/ExperimentDetails.tsx
// COMPLETE FILE WITH ALL TASK-AWARE UPDATES + LLM JUDGE FIXES APPLIED
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, RefreshCw, Trash2, Loader2, CheckCircle2, FileText } from 'lucide-react';
import toast from 'react-hot-toast';
import { api } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';
import type { Experiment, ModelVariant, ComparativeMetrics } from '../types';
import { BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { GenerationProgress } from '../components/GenerationProgress';


const renderTaskSpecificMetrics = (
  metrics: ComparativeMetrics | undefined,
  taskType: string | undefined
) => {
  if (!metrics) {
    return <p className="text-gray-500 text-sm">Waiting for metrics...</p>;
  }

  const results = metrics.evaluation_results;

  // ALWAYS show performance metrics (hardware stats)
  const performanceSection = (
    <div className="grid grid-cols-2 gap-2 mt-3">
      {/* Speed */}
      {metrics.avg_tokens_per_second && (
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-3 border border-blue-200">
          <div className="flex items-center justify-between">
            <span className="text-xs text-blue-700 font-medium">⚡ Speed</span>
            <span className="text-lg font-bold text-blue-900">
              {metrics.avg_tokens_per_second.toFixed(1)}
            </span>
          </div>
          <p className="text-xs text-blue-600 mt-1">tok/s</p>
        </div>
      )}
      
      {/* Latency */}
      {metrics.avg_latency_ms && (
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-3 border border-purple-200">
          <div className="flex items-center justify-between">
            <span className="text-xs text-purple-700 font-medium">⏱️ Latency</span>
            <span className="text-lg font-bold text-purple-900">
              {(metrics.avg_latency_ms / 1000).toFixed(1)}
            </span>
          </div>
          <p className="text-xs text-purple-600 mt-1">seconds</p>
        </div>
      )}
      
      {/* Model Size */}
      {metrics.model_size_mb && (
        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-3 border border-green-200">
          <div className="flex items-center justify-between">
            <span className="text-xs text-green-700 font-medium">💾 Size</span>
            <span className="text-lg font-bold text-green-900">
              {(metrics.model_size_mb / 1024).toFixed(2)}
            </span>
          </div>
          <p className="text-xs text-green-600 mt-1">GB</p>
        </div>
      )}
      
      {/* Avg Tokens */}
      {metrics.avg_token_count && (
        <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 rounded-lg p-3 border border-yellow-200">
          <div className="flex items-center justify-between">
            <span className="text-xs text-yellow-700 font-medium">📝 Tokens</span>
            <span className="text-lg font-bold text-yellow-900">
              {metrics.avg_token_count.toFixed(0)}
            </span>
          </div>
          <p className="text-xs text-yellow-600 mt-1">avg/response</p>
        </div>
      )}
    </div>
  );

  // If no evaluation results yet, just show performance
  if (!results) {
    return performanceSection;
  }

  // TEXT GENERATION METRICS
  if (taskType === 'text_generation') {
    return (
      <div className="space-y-4">
        {/* Quality Score with Visual Bar */}
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 border border-indigo-200">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-semibold text-gray-700">🎯 Quality (BERTScore)</span>
            <span className="text-2xl font-bold text-indigo-600">
              {((results.bertscore_f1 || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${
                (results.bertscore_f1 || 0) >= 0.7
                  ? 'bg-gradient-to-r from-green-400 to-green-600'
                  : (results.bertscore_f1 || 0) >= 0.5
                  ? 'bg-gradient-to-r from-yellow-400 to-yellow-600'
                  : 'bg-gradient-to-r from-red-400 to-red-600'
              }`}
              style={{ width: `${((results.bertscore_f1 || 0) * 100)}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {(results.bertscore_f1 || 0) >= 0.7 ? '✅ Excellent quality' : 
             (results.bertscore_f1 || 0) >= 0.5 ? '⚠️ Acceptable quality' : 
             '❌ Poor quality'}
          </p>
        </div>

        {/* Length & Divergence */}
        <div className="grid grid-cols-2 gap-3">
          {/* Length Ratio */}
          <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
            <p className="text-xs text-blue-700 font-medium mb-1">📏 Length Match</p>
            <p className="text-xl font-bold text-blue-900">
              {(results.length_ratio_mean || 1).toFixed(1)}x
            </p>
            <p className="text-xs text-blue-600 mt-1">
              {(results.length_ratio_mean || 1) > 2 ? '⚠️ Too verbose' : 
               (results.length_ratio_mean || 1) < 0.5 ? '⚠️ Too short' : 
               '✅ Good length'}
            </p>
          </div>

          {/* Divergence */}
          {results.divergence_score !== undefined && results.divergence_score !== null ? (
            <div className="bg-orange-50 rounded-lg p-3 border border-orange-200">
              <p className="text-xs text-orange-700 font-medium mb-1">🔀 Divergence</p>
              <p className="text-xl font-bold text-orange-900">
                {((results.divergence_score || 0) * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-orange-600 mt-1">from baseline</p>
            </div>
          ) : (
            <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
              <p className="text-xs text-gray-500 font-medium mb-1">🔀 Divergence</p>
              <p className="text-sm text-gray-400 mt-2">N/A (no baseline)</p>
            </div>
          )}
        </div>

        {/* Precision/Recall Details (Collapsible) */}
        {results.bertscore_precision && (
          <details className="bg-gray-50 rounded-lg p-3 border border-gray-200">
            <summary className="cursor-pointer text-xs font-medium text-gray-700 hover:text-gray-900">
              📊 Show detailed metrics
            </summary>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-600">Precision:</span>
                <span className="font-semibold">{((results.bertscore_precision || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Recall:</span>
                <span className="font-semibold">{((results.bertscore_recall || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Length Std:</span>
                <span className="font-semibold">±{(results.length_ratio_std || 0).toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Reference:</span>
                <span className="font-semibold">{results.reference_type || 'unknown'}</span>
              </div>
            </div>
          </details>
        )}

        {/* ✅ FIXED: LLM Judge for Text Generation */}
        {results.llm_judge && (
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-4 border border-purple-200">
            <p className="text-sm font-semibold text-purple-900 mb-3">🤖 LLM Judge Results</p>
            
            {/* Display accuracy, fluency, coherence (1-5 scale) */}
            <div className="grid grid-cols-3 gap-3">
              {results.llm_judge.avg_accuracy !== undefined && (
                <div className="text-center">
                  <p className="text-xs text-purple-600 mb-1">Accuracy</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {results.llm_judge.avg_accuracy.toFixed(1)}
                  </p>
                  <p className="text-xs text-purple-500">out of 5</p>
                </div>
              )}
              {results.llm_judge.avg_fluency !== undefined && (
                <div className="text-center">
                  <p className="text-xs text-purple-600 mb-1">Fluency</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {results.llm_judge.avg_fluency.toFixed(1)}
                  </p>
                  <p className="text-xs text-purple-500">out of 5</p>
                </div>
              )}
              {results.llm_judge.avg_coherence !== undefined && (
                <div className="text-center">
                  <p className="text-xs text-purple-600 mb-1">Coherence</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {results.llm_judge.avg_coherence.toFixed(1)}
                  </p>
                  <p className="text-xs text-purple-500">out of 5</p>
                </div>
              )}
            </div>
            
            {/* Sample info */}
            {results.llm_judge.num_evaluated && (
              <p className="text-xs text-purple-600 mt-3 text-center">
                📊 Evaluated {results.llm_judge.num_evaluated} samples 
                ({results.llm_judge.sample_percentage || 10}% of dataset)
              </p>
            )}
            
            {/* Show error if judge failed */}
            {results.llm_judge.error && (
              <p className="text-xs text-red-600 mt-2">
                ⚠️ Judge error: {results.llm_judge.error}
              </p>
            )}
          </div>
        )}

        {/* Performance Section */}
        {performanceSection}
      </div>
    );
  }

  // CLASSIFICATION METRICS
  if (taskType === 'classification') {
    return (
      <div className="space-y-4">
        {/* Accuracy */}
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4 border border-green-200">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-semibold text-gray-700">🎯 Accuracy</span>
            <span className="text-2xl font-bold text-green-600">
              {((results.accuracy || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-green-400 to-green-600 h-3 rounded-full transition-all"
              style={{ width: `${((results.accuracy || 0) * 100)}%` }}
            />
          </div>
        </div>

        {/* F1 Scores */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
            <p className="text-xs text-blue-700 font-medium mb-1">📊 Macro F1</p>
            <p className="text-xl font-bold text-blue-900">
              {((results.macro_f1 || 0) * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-purple-50 rounded-lg p-3 border border-purple-200">
            <p className="text-xs text-purple-700 font-medium mb-1">⚖️ Weighted F1</p>
            <p className="text-xl font-bold text-purple-900">
              {((results.weighted_f1 || 0) * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Class Imbalance Warning */}
        {results.is_imbalanced && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
            <p className="text-sm text-yellow-800">
              ⚠️ <strong>Class imbalance detected</strong> - weighted F1 may be more reliable
            </p>
          </div>
        )}

        {/* Per-Class F1 */}
        {results.per_class_f1 && (
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <p className="text-sm font-semibold text-gray-700 mb-3">Per-Class F1 Scores</p>
            <div className="space-y-2">
              {Object.entries(results.per_class_f1).map(([className, f1]) => (
                <div key={className}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">{className}</span>
                    <span className="font-semibold">{((f1 as number) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${((f1 as number) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Performance Section */}
        {performanceSection}
      </div>
    );
  }

  // RAG METRICS
  if (taskType === 'rag') {
    return (
      <div className="space-y-4">
        {/* Answer Relevance */}
        <div className="bg-gradient-to-r from-cyan-50 to-blue-50 rounded-lg p-4 border border-cyan-200">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-semibold text-gray-700">🔍 Answer Relevance</span>
            <span className="text-2xl font-bold text-cyan-600">
              {((results.answer_relevance || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-cyan-400 to-cyan-600 h-3 rounded-full transition-all"
              style={{ width: `${((results.answer_relevance || 0) * 100)}%` }}
            />
          </div>
        </div>

        {/* BERTScore (if available) */}
        {results.bertscore_f1 && (
          <div className="bg-indigo-50 rounded-lg p-3 border border-indigo-200">
            <p className="text-xs text-indigo-700 font-medium mb-1">🎯 BERTScore</p>
            <p className="text-xl font-bold text-indigo-900">
              {((results.bertscore_f1 || 0) * 100).toFixed(1)}%
            </p>
          </div>
        )}

        {/* ✅ FIXED: LLM Judge - Hallucination with correct scale */}
        {results.llm_judge && (
          <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-lg p-4 border border-red-200">
            <p className="text-sm font-semibold text-red-900 mb-3">🤖 Hallucination Check</p>
            
            {results.llm_judge.hallucination_rate !== undefined && (
              <div className="mb-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-red-700">Hallucination Rate</span>
                  <span className="text-2xl font-bold text-red-900">
                    {results.llm_judge.hallucination_rate.toFixed(1)}%  {/* ✅ NO * 100 */}
                  </span>
                </div>
                {/* Visual bar */}
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-red-500 h-2 rounded-full"
                    style={{ width: `${Math.min(100, results.llm_judge.hallucination_rate)}%` }}
                  />
                </div>
                <p className="text-xs text-red-600 mt-1">
                  {results.llm_judge.hallucination_rate < 10 ? '✅ Low hallucination' :
                   results.llm_judge.hallucination_rate < 30 ? '⚠️ Moderate hallucination' :
                   '❌ High hallucination'}
                </p>
              </div>
            )}
            
            {results.llm_judge.avg_factual_correctness !== undefined && (
              <div className="mb-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-orange-700">Factual Correctness</span>
                  <span className="text-xl font-bold text-orange-900">
                    {results.llm_judge.avg_factual_correctness.toFixed(1)}/5
                  </span>
                </div>
              </div>
            )}
            
            {results.llm_judge.avg_completeness !== undefined && (
              <div className="mb-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-orange-700">Completeness</span>
                  <span className="text-xl font-bold text-orange-900">
                    {results.llm_judge.avg_completeness.toFixed(1)}/5
                  </span>
                </div>
              </div>
            )}
            
            {results.llm_judge.num_evaluated && (
              <p className="text-xs text-red-600 mt-2 text-center">
                📊 Evaluated {results.llm_judge.num_evaluated} samples
                ({results.llm_judge.sample_percentage || 10}% of dataset)
              </p>
            )}
            
            {results.llm_judge.error && (
              <p className="text-xs text-red-600 mt-2">
                ⚠️ Judge error: {results.llm_judge.error}
              </p>
            )}
          </div>
        )}

        {/* Performance Section */}
        {performanceSection}
      </div>
    );
  }

  // FALLBACK: Just show performance
  return performanceSection;
};


export const ExperimentDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const experimentId = parseInt(id || '0');

  const [evaluating, setEvaluating] = useState(false);
  const [enableLLMJudge, setEnableLLMJudge] = useState(false);
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
    refetchInterval: autoRefreshEnabled ? 3000 : false,
  });

  // Fetch variants
  const { 
    data: variants = [], 
    isLoading: isLoadingVariants,
    refetch: refetchVariants 
  } = useQuery<ModelVariant[]>({
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
  } = useQuery<ComparativeMetrics[]>({
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

  const needsEvaluation = experiment?.status === 'completed' && metrics.length === 0;

  // Handle evaluation with LLM judge
  const handleEvaluation = async () => {
    setEvaluating(true);

    const toastId = toast.loading('🧮 Evaluation started! Calculating metrics...', {
      duration: Infinity,
    });

    try {
      await api.triggerEvaluation(experimentId, enableLLMJudge);

      let attempts = 0;
      const maxAttempts = 40;

      const pollInterval = setInterval(async () => {
        attempts++;
        
        const newMetrics = await refetchMetrics();
        
        if (newMetrics.data && newMetrics.data.length > 0) {
          clearInterval(pollInterval);
          setEvaluating(false);
          toast.success('✅ Evaluation complete!', { id: toastId });
          refetchVariants();
          refetchExperiment();
        } else if (attempts >= maxAttempts) {
          clearInterval(pollInterval);
          setEvaluating(false);
          toast.error('⚠️ Evaluation taking longer than expected.', { id: toastId });
        }
      }, 3000);

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

  const baselineVariant = variants.find(v => v.variant_type === 'baseline');
  const quantizedVariants = variants.filter(v => v.variant_type === 'quantized');

  const getMetricsForVariant = (variantId: number) => {
    return metrics.find(m => m.variant_id === variantId);
  };

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

  const isGenerating = experiment.status === 'generating' || 
                       variants.some(v => v.status === 'generating');

  return (
    <div className="space-y-6">
      {/* Header with task type badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link to="/" className="text-gray-600 hover:text-gray-900 transition-colors">
            <ArrowLeft className="w-6 h-6" />
          </Link>
          <div>
            <div className="flex items-center space-x-3">
              <h1 className="text-3xl font-bold text-gray-900">{experiment.name}</h1>
              {experiment.task_type && (
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-800">
                  {experiment.task_type === 'text_generation' && '🔤 Text Generation'}
                  {experiment.task_type === 'classification' && '🎯 Classification'}
                  {experiment.task_type === 'rag' && '🔍 RAG'}
                </span>
              )}
            </div>
            <p className="text-gray-600">{experiment.sample_count} samples</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex flex-col items-end space-y-2">
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-500 uppercase tracking-wide font-medium">
                Generation
              </span>
              <StatusBadge status={experiment.status} />
            </div>

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

      {isGenerating && (
        <GenerationProgress
          experimentId={experimentId}
          onComplete={() => {
            // Refresh data when generation completes
            refetchExperiment();
            refetchVariants();
            refetchMetrics();
            
            // Optionally show a success message
            toast.success('✅ Generation complete! Ready for evaluation.');
          }}
        />
      )}

      {/* Evaluation section with LLM judge toggle */}
      {needsEvaluation && !isGenerating && (
        <div className="card bg-yellow-50 border-yellow-200">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-yellow-800 font-medium">
                ⚠️ Generation complete. Run evaluation to see metrics.
              </p>
              <p className="text-sm text-yellow-700 mt-1">
                Evaluation takes 60-90 seconds
              </p>
              {(experiment.task_type === 'text_generation' || experiment.task_type === 'rag') && (
                <label className="flex items-center space-x-2 mt-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableLLMJudge}
                    onChange={(e) => setEnableLLMJudge(e.target.checked)}
                    className="rounded text-primary-600"
                  />
                  <span className="text-sm text-gray-700">
                    Enable LLM Judge (slower, requires GROQ_API_KEY)
                  </span>
                </label>
              )}
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

        {/* Quantized Cards with task-specific metrics */}
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

              {renderTaskSpecificMetrics(variantMetrics as ComparativeMetrics, experiment.task_type)}
            </div>
          );
        })}
      </div>

      {/* VISUALIZATION SECTION - GRAPHS */}
      {metrics.length > 0 && quantizedVariants.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center space-x-2 mb-4">
            <span className="text-2xl">📊</span>
            <h2 className="text-2xl font-bold text-gray-900">Performance Analysis</h2>
          </div>

          {/* Graph 1: Quality Comparison (Bar Chart) */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              🎯 Quality Comparison (
                {experiment.task_type === 'classification' ? 'Accuracy' :
                experiment.task_type === 'rag' ? 'Answer Relevance' :
                'BERTScore'}
              )
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={[
                  ...(baselineVariant ? [{
                    name: 'Baseline',
                    quality: 100,
                    fill: '#3b82f6'
                  }] : []),
                  ...quantizedVariants.map((variant) => {
                    const variantMetrics = getMetricsForVariant(variant.id);
                    let score = 0;
                    if (experiment.task_type === 'classification') {
                      score = variantMetrics?.evaluation_results?.accuracy || 0;
                    } else if (experiment.task_type === 'rag') {
                      score = variantMetrics?.evaluation_results?.answer_relevance || 0;
                    } else {
                      score = variantMetrics?.evaluation_results?.bertscore_f1 || 0;
                    }
                    return {
                      name: variant.model_name,
                      quality: score * 100,
                      fill: score >= 0.7 ? '#10b981' : score >= 0.5 ? '#f59e0b' : '#ef4444'
                    };
                  })
                ]}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Quality (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Bar dataKey="quality" radius={[8, 8, 0, 0]}>
                  {quantizedVariants.map((_, index) => (
                    <Cell key={`cell-${index}`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-500 mt-2 text-center">
              Higher is better • Green = Excellent (70%+) • Yellow = Acceptable (50-70%) • Red = Poor (&lt;50%)
            </p>
          </div>

          {/* Graph 2: Size vs Quality (Scatter Plot) */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              💾 Size vs Quality Trade-off
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  type="number" 
                  dataKey="size" 
                  name="Size" 
                  label={{ value: 'Model Size (GB)', position: 'bottom' }}
                  domain={[0, 'auto']}
                />
                <YAxis 
                  type="number" 
                  dataKey="quality" 
                  name="Quality" 
                  label={{ value: 'Quality (%)', angle: -90, position: 'insideLeft' }}
                  domain={[0, 100]}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ payload }) => {
                    if (payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
                          <p className="font-semibold">{data.name}</p>
                          <p className="text-sm">Size: {data.size.toFixed(2)} GB</p>
                          <p className="text-sm">Quality: {data.quality.toFixed(1)}%</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter 
                  name="Models" 
                  data={quantizedVariants.map((variant) => {
                    const variantMetrics = getMetricsForVariant(variant.id);
                    
                    let quality = 0;
                    if (experiment.task_type === 'classification') {
                      quality = (variantMetrics?.evaluation_results?.accuracy || 0) * 100;
                    } else if (experiment.task_type === 'rag') {
                      quality = (variantMetrics?.evaluation_results?.answer_relevance || 0) * 100;
                    } else {
                      quality = (variantMetrics?.evaluation_results?.bertscore_f1 || 0) * 100;
                    }
                    
                    return {
                      name: variant.model_name,
                      size: (variantMetrics?.model_size_mb || 0) / 1024,
                      quality: quality
                    };
                  })}
                  fill="#8884d8"
                  shape="circle"
                >
                  {quantizedVariants.map((_, index) => (
                    <Cell key={`cell-${index}`} fill="#6366f1" />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-500 mt-2 text-center">
              Ideal models: Top-left corner (small size, high quality)
            </p>
          </div>
        </div>
      )}

      {/* Comparison Charts */}
      {metrics.length > 0 && baselineVariant && quantizedVariants.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Speed Comparison</h3>
            <div className="space-y-4">
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

          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quality Comparison</h3>
            <div className="space-y-4">
              {quantizedVariants.map((variant) => {
                const variantMetrics = getMetricsForVariant(variant.id);
                let quality = 0;
                if (experiment.task_type === 'classification') {
                  quality = (variantMetrics?.evaluation_results?.accuracy || 0) * 100;
                } else if (experiment.task_type === 'rag') {
                  quality = (variantMetrics?.evaluation_results?.answer_relevance || 0) * 100;
                } else {
                  quality = (variantMetrics?.evaluation_results?.bertscore_f1 || 0) * 100;
                }
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
        <div className="card bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <span className="mr-2">💡</span> What This Means
          </h3>
          
          {quantizedVariants.map((variant, index) => {
            const variantMetrics = getMetricsForVariant(variant.id);
            const baselineMetrics = getMetricsForVariant(baselineVariant.id);
            const baselineSpeed = baselineMetrics?.avg_tokens_per_second || 0;
            const quantSpeed = variantMetrics?.avg_tokens_per_second || 0;
            const speedRatio = baselineSpeed / quantSpeed;
            
            let quality = 0;
            if (experiment.task_type === 'classification') {
              quality = (variantMetrics?.evaluation_results?.accuracy || 0) * 100;
            } else if (experiment.task_type === 'rag') {
              quality = (variantMetrics?.evaluation_results?.answer_relevance || 0) * 100;
            } else {
              quality = (variantMetrics?.evaluation_results?.bertscore_f1 || 0) * 100;
            }
            
            const divergence = variantMetrics?.evaluation_results?.divergence_score;
            const sizeGB = (variantMetrics?.model_size_mb || 0) / 1024;
            const lengthRatio = variantMetrics?.evaluation_results?.length_ratio_mean || 1;

            const qualityVerdict = quality >= 70 ? "excellent" : quality >= 50 ? "acceptable" : "poor";
            const speedVerdict = speedRatio > 50 ? "much slower" : speedRatio > 10 ? "significantly slower" : "somewhat slower";
            const lengthVerdict = lengthRatio > 2 ? "much more verbose" : lengthRatio > 1.5 ? "more detailed" : lengthRatio < 0.8 ? "more concise" : "similar length";
            
            return (
              <div key={variant.id} className={index > 0 ? 'mt-4 pt-4 border-t border-blue-200' : ''}>
                <h4 className="font-semibold text-gray-800 mb-3">
                  {variant.model_name} • {variant.quantization_level}
                </h4>
                
                <div className="space-y-3 text-gray-700">
                  <p className="text-base leading-relaxed">
                    This quantized model delivers <strong className="text-gray-900">{qualityVerdict} quality</strong> at {quality.toFixed(0)}%, 
                    while being <strong className="text-gray-900">{speedVerdict}</strong> than the cloud baseline 
                    ({speedRatio.toFixed(1)}x difference). The model runs entirely on your device with just <strong className="text-gray-900">{sizeGB.toFixed(2)} GB</strong> of storage.
                  </p>

                  {lengthRatio && (
                    <p className="text-sm leading-relaxed">
                      📝 <strong>Response style:</strong> This model generates {lengthVerdict} responses compared to the reference 
                      ({lengthRatio.toFixed(1)}x {lengthRatio > 1 ? 'longer' : lengthRatio < 1 ? 'shorter' : 'similar'}).
                      {lengthRatio > 2 && " ⚠️ Not ideal if you need brief, concise outputs."}
                      {lengthRatio > 1 && lengthRatio <= 2 && " ✅ Good if you prefer detailed explanations."}
                      {lengthRatio < 0.8 && " ✅ Perfect if you prefer brief, to-the-point responses."}
                      {lengthRatio >= 0.8 && lengthRatio <= 1.2 && " ✅ Response length matches the reference well."}
                    </p>
                  )}

                  {divergence !== undefined && divergence !== null ? (
                    <p className="text-sm leading-relaxed">
                      🔀 <strong>Output similarity:</strong> The model's responses differ by {(divergence * 100).toFixed(0)}% from the baseline, 
                      {divergence < 0.1 ? " meaning it stays very close to the reference behavior." : 
                      divergence < 0.3 ? " showing moderate variation in phrasing while maintaining meaning." : 
                      " indicating significant differences in how it expresses ideas."}
                    </p>
                  ) : (
                    <p className="text-sm leading-relaxed text-gray-500">
                      ℹ️ <strong>Divergence unavailable:</strong> This metric requires comparing against a baseline model's outputs.
                    </p>
                  )}

                  <div className="bg-white/50 rounded-lg p-3 mt-3">
                    <p className="text-sm font-medium text-gray-800">
                      {quality >= 70 && sizeGB < 3 && "✅ Great choice! High quality with reasonable size." }
                      {quality >= 70 && sizeGB >= 3 && "⚖️ High quality but larger size. Worth it if storage isn't a concern." }
                      {quality >= 50 && quality < 70 && "⚠️ Acceptable trade-off if you need smaller size or faster local inference." }
                      {quality < 50 && "❌ Quality may be too low for production use. Consider a larger quantization level." }
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