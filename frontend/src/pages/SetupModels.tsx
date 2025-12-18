// frontend/src/pages/SetupModels.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { ArrowLeft, Loader2, CheckCircle2 } from 'lucide-react';
import toast from 'react-hot-toast';
import { api } from '../services/api';

interface ModelOption {
  id: string;
  name: string;
  description: string;
  speed?: string;
  quality?: string;
  size_mb?: number;
  filename?: string;
  quantization?: string;
}

export const SetupModels: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const experimentId = parseInt(id || '0');

  const [loading, setLoading] = useState(false);
  const [isNavigating, setIsNavigating] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [hasGroundTruth, setHasGroundTruth] = useState(false);
  const [baselineModels, setBaselineModels] = useState<ModelOption[]>([]);
  const [quantizedModels, setQuantizedModels] = useState<ModelOption[]>([]);
  
  const [selectedBaseline, setSelectedBaseline] = useState<string | null>('llama-3.3-70b-versatile');
  const [selectedQuantized, setSelectedQuantized] = useState<string[]>([]);
  
  // Real-time generation progress state
  const [generationProgress, setGenerationProgress] = useState<{
    overall: number;
    variants: Array<{ name: string; progress: number; completed: number; total: number }>;
    allCompleted: boolean;
    totalCompleted: number;
    totalSamples: number;
  } | null>(null);

  // Load experiment details and available models
  useEffect(() => {
    const loadModels = async () => {
      try {
        // Get experiment to check if has ground truth
        const expRes = await fetch(`http://localhost:8000/experiments/${experimentId}`);
        const expData = await expRes.json();
        setHasGroundTruth(expData.has_ground_truth);
        
        // ✅ FIX: If experiment is already generating or completed, redirect immediately
        if (expData.status === 'generating' || expData.status === 'completed') {
          navigate(`/experiment/${experimentId}`, { replace: true });
          return;
        }
        
        const [baselineRes, quantizedRes] = await Promise.all([
          fetch('http://localhost:8000/experiments/models/baseline'),
          fetch('http://localhost:8000/experiments/models/quantized')
        ]);
        
        const baselineData = await baselineRes.json();
        const quantizedData = await quantizedRes.json();
        
        setBaselineModels(baselineData.models);
        setQuantizedModels(quantizedData.models);
        
        // If has ground truth, baseline is optional
        if (expData.has_ground_truth) {
          setSelectedBaseline(null);
        }
      } catch (error) {
        toast.error('Failed to load available models');
      } finally {
        setLoadingModels(false);
      }
    };

    loadModels();
  }, [experimentId]);

  // Poll for generation progress when loading (generation in progress)
  useEffect(() => {
    if (!loading) return;

    let intervalId: number | null = null;
    let isNavigatingAway = false;
    
    const fetchProgress = async () => {
      // Don't fetch if we're already navigating away
      if (isNavigatingAway) return;
      
      try {
        const response = await fetch(
          `http://localhost:8000/experiments/${experimentId}/generation-status?t=${Date.now()}`
        );
        
        if (!response.ok) return;
        
        const data = await response.json();
        
        // Get progress from the first variant (they all process the same samples)
        // This shows sample-level progress: "1/14, 2/14, 3/14" etc.
        const firstVariant = data.variants.length > 0 ? data.variants[0] : null;
        const totalCompleted = firstVariant?.completed_samples || 0;
        const totalSamples = firstVariant?.total_samples || 0;
        
        // Calculate overall progress (0.0 to 1.0) based on actual sample count
        const overallProgress = totalSamples > 0 ? totalCompleted / totalSamples : 0;
        
        setGenerationProgress({
          overall: overallProgress,
          variants: data.variants.map((v: any) => ({
            name: v.model_name,
            progress: v.progress || 0,
            completed: v.completed_samples || 0,
            total: v.total_samples || 0
          })),
          allCompleted: data.all_completed,
          totalCompleted: totalCompleted,
          totalSamples: totalSamples
        });

        // If all completed, navigate away after a short delay
        if (data.all_completed && !isNavigatingAway) {
          isNavigatingAway = true;
          if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
          }
          setTimeout(() => {
            setLoading(false);
            setIsNavigating(true);
            navigate(`/experiment/${experimentId}`, { replace: true });
          }, 1000);
        }
      } catch (err) {
        console.error('Error fetching progress:', err);
      }
    };

    // Start polling immediately, then every 1 second
    fetchProgress();
    intervalId = setInterval(fetchProgress, 1000);

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [loading, experimentId, navigate]);

  // Validation: Check if selection is valid
  const isValid = () => {
    if (hasGroundTruth) {
      // With ground truth: at least 1 model (baseline OR quantized)
      return selectedBaseline !== null || selectedQuantized.length > 0;
    } else {
      // Without ground truth: need both baseline AND quantized
      return selectedBaseline !== null && selectedQuantized.length > 0;
    }
  };
  
    const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log("🔍 DEBUG: experimentId =", experimentId);
    console.log("🔍 DEBUG: selectedBaseline =", selectedBaseline);
    console.log("🔍 DEBUG: selectedQuantized =", selectedQuantized);
    
    if (!isValid()) {
      toast.error('Please select required models');
      return;
    }
    
    setLoading(true);
    const toastId = toast.loading('🚀 Creating experiment variants...');

    try {
      // Create baseline variant if selected
      if (selectedBaseline) {
        await api.createVariant(experimentId, {
          variant_type: 'baseline',
          model_name: selectedBaseline,
          inference_provider: 'groq',
        });
      }

      // Create quantized variants
      if (selectedQuantized.length > 0) {
        toast.loading('⚙️ Setting up quantized models...', { id: toastId });
        
        for (const quantId of selectedQuantized) {
          const quantModel = quantizedModels.find(m => m.id === quantId);
          if (quantModel) {
            await api.createVariant(experimentId, {
              variant_type: 'quantized',
              model_name: quantId,
              quantization_level: quantModel.quantization || 'INT4',
              model_path: `data/models/quantized/${quantModel.filename}`,
              inference_provider: 'gguf',
            });
          }
        }
      }

      toast.loading('🎯 Starting generation...', { id: toastId });

      // Trigger generation
      await api.triggerGeneration(experimentId);

      // Keep modal visible and start polling for progress
      // Don't navigate away - show progress in the modal

    } catch (error) {
      console.error('Failed:', error);
      toast.error('❌ Failed to start generation: ' + error, { id: toastId });
      setLoading(false);
    }
  };

  const toggleQuantized = (modelId: string) => {
    if (selectedQuantized.includes(modelId)) {
      setSelectedQuantized(selectedQuantized.filter(id => id !== modelId));
    } else {
      setSelectedQuantized([...selectedQuantized, modelId]);
    }
  };

  const calculateTotalSize = () => {
    return selectedQuantized.reduce((total, id) => {
      const model = quantizedModels.find(m => m.id === id);
      return total + (model?.size_mb || 0);
    }, 0);
  };

  // Loading state
  if (loadingModels) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading available models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Link 
          to={`/experiment/${experimentId}/upload`} 
          className="text-gray-600 hover:text-gray-900 transition-colors"
        >
          <ArrowLeft className="w-6 h-6" />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Select Models</h1>
          <p className="text-gray-600">
            {hasGroundTruth 
              ? 'Choose models to compare against ground truth'
              : 'Choose baseline and quantized models to compare'
            }
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Baseline Model Selection */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">🌟 Baseline Model</h3>
            <div className="flex items-center space-x-2">
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                Cloud API
              </span>
              {hasGroundTruth && (
                <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
                  Optional
                </span>
              )}
              {!hasGroundTruth && (
                <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded">
                  Required
                </span>
              )}
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            {hasGroundTruth 
              ? 'Optional - will be compared against ground truth if selected'
              : 'Required - will serve as quality reference for quantized models'
            }
          </p>
          
          {/* Skip Baseline Option (only if has ground truth) */}
          {hasGroundTruth && (
            <label className="block p-4 border-2 rounded-lg cursor-pointer mb-2 transition-all border-gray-200 hover:border-gray-300 hover:shadow-sm">
              <input
                type="radio"
                name="baseline"
                checked={selectedBaseline === null}
                onChange={() => setSelectedBaseline(null)}
                disabled={loading}
                className="sr-only"
              />
              <div className="flex items-center space-x-3">
                {selectedBaseline === null && (
                  <CheckCircle2 className="w-5 h-5 text-primary-600 flex-shrink-0" />
                )}
                <span className="font-semibold text-gray-900">
                  Skip baseline (compare only against ground truth)
                </span>
              </div>
            </label>
          )}
          
          {/* Baseline Model Options */}
          <div className="space-y-2">
            {baselineModels.map((model) => (
              <label
                key={model.id}
                className={`block p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  selectedBaseline === model.id
                    ? 'border-primary-500 bg-primary-50 shadow-sm'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
                }`}
              >
                <input
                  type="radio"
                  name="baseline"
                  value={model.id}
                  checked={selectedBaseline === model.id}
                  onChange={(e) => setSelectedBaseline(e.target.value)}
                  className="sr-only"
                  disabled={loading}
                />
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    {selectedBaseline === model.id && (
                      <CheckCircle2 className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
                    )}
                    <div>
                      <p className="font-semibold text-gray-900">{model.name}</p>
                      <p className="text-sm text-gray-600">{model.description}</p>
                    </div>
                  </div>
                  <div className="flex flex-col items-end space-y-1">
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded whitespace-nowrap">
                      {model.speed}
                    </span>
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded whitespace-nowrap">
                      {model.quality}
                    </span>
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Quantized Model Selection */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">⚡ Quantized Models</h3>
            <div className="flex items-center space-x-2">
              <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                Local Inference
              </span>
              {!hasGroundTruth && (
                <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded">
                  Required
                </span>
              )}
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            Select one or more quantized models • {selectedQuantized.length} selected
          </p>
          
          <div className="space-y-2">
            {quantizedModels.map((model) => (
              <label
                key={model.id}
                className={`block p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  selectedQuantized.includes(model.id)
                    ? 'border-green-500 bg-green-50 shadow-sm'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedQuantized.includes(model.id)}
                  onChange={() => toggleQuantized(model.id)}
                  className="sr-only"
                  disabled={loading}
                />
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    {selectedQuantized.includes(model.id) && (
                      <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                    )}
                    <div>
                      <p className="font-semibold text-gray-900">{model.name}</p>
                      <p className="text-sm text-gray-600">{model.description}</p>
                      {model.size_mb && (
                        <p className="text-xs text-gray-500 mt-1">
                          📦 Size: {model.size_mb}MB
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex flex-col items-end space-y-1">
                    {model.quantization && (
                      <span className="text-xs bg-teal-100 text-teal-700 px-2 py-1 rounded whitespace-nowrap font-semibold">
                        {model.quantization}
                      </span>
                    )}
                    <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded whitespace-nowrap">
                      {model.speed}
                    </span>
                    <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded whitespace-nowrap">
                      {model.quality}
                    </span>
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Summary Info */}
        <div className="card bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <span className="text-xl">📊</span>
              </div>
            </div>
            <div className="flex-1">
              <p className="font-medium text-gray-900 mb-1">Experiment Summary</p>
              <ul className="text-sm text-gray-700 space-y-1">
                {hasGroundTruth && (
                  <li className="flex items-center">
                    <span className="mr-2">✅</span>
                    Ground truth provided - baseline is optional
                  </li>
                )}
                {!hasGroundTruth && (
                  <li className="flex items-center">
                    <span className="mr-2">⚠️</span>
                    No ground truth - baseline required as reference
                  </li>
                )}
                {selectedBaseline && (
                  <li>
                    • 1 baseline model ({baselineModels.find(m => m.id === selectedBaseline)?.name})
                  </li>
                )}
                {selectedBaseline === null && hasGroundTruth && (
                  <li>
                    • No baseline model (skipped)
                  </li>
                )}
                <li>
                  • {selectedQuantized.length} quantized model{selectedQuantized.length !== 1 ? 's' : ''}
                </li>
                {selectedQuantized.length > 0 && (
                  <li>
                    • Total model size: ~{calculateTotalSize()}MB
                  </li>
                )}
                <li>
                  • Generation will start automatically after setup
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Validation Message */}
        {!isValid() && (
          <div className="card bg-yellow-50 border-yellow-200">
            <p className="text-sm text-yellow-800">
              {hasGroundTruth 
                ? '⚠️ Please select at least one model (baseline or quantized)'
                : '⚠️ Please select at least one baseline model and one quantized model'
              }
            </p>
          </div>
        )}

        {/* Buttons */}
        <div className="flex justify-between items-center">
          <Link 
            to={`/experiment/${experimentId}/upload`} 
            className={`btn-secondary ${loading ? 'pointer-events-none opacity-50' : ''}`}
          >
            Back
          </Link>
          <button
            type="submit"
            disabled={loading || !isValid()}
            className="btn-primary flex items-center space-x-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Starting Generation...</span>
              </>
            ) : (
              <>
                <span>🚀 Start Generation</span>
                <span className="text-xs opacity-75">
                  ({[selectedBaseline, ...selectedQuantized].filter(Boolean).length} model{[selectedBaseline, ...selectedQuantized].filter(Boolean).length !== 1 ? 's' : ''})
                </span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Loading Overlay - hide if navigating */}
      {loading && !isNavigating && (
        <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-sm w-full mx-4">
            <div className="text-center" key={`progress-${generationProgress?.totalCompleted || 0}`}>
              <Loader2 className="w-12 h-12 animate-spin text-primary-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Setting Up Experiment
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                {generationProgress 
                  ? `Generating outputs... ${generationProgress.totalCompleted}/${generationProgress.totalSamples} samples`
                  : 'Creating variants and starting generation...'
                }
              </p>
              
              {/* Real-time progress bar */}
              <div className="mt-4 w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div 
                  className="bg-primary-600 h-2 rounded-full transition-all duration-500 ease-out" 
                  style={{ 
                    width: `${Math.max(0, Math.min(100, (generationProgress?.overall || 0) * 100))}%` 
                  }} 
                />
              </div>
              
              {/* Show percentage and sample count */}
              {generationProgress && (
                <p className="text-xs text-gray-500 mt-2">
                  {generationProgress.totalCompleted}/{generationProgress.totalSamples} samples • {Math.round(generationProgress.overall * 100)}% complete
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};