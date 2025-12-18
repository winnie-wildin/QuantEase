// frontend/src/components/GenerationProgress.tsx
import React, { useState, useEffect } from 'react';
import { AlertCircle, X } from 'lucide-react';
import { ProgressBar } from './ProgressBar';
import { api } from '../services/api';
import toast from 'react-hot-toast';

interface RecentSample {
  position: number;
  output_preview: string;
  latency_ms: number;
  is_successful: boolean;
  timestamp: number;
}

interface VariantStatus {
  variant_id: number;
  model_name: string;
  variant_type: string;
  status: string;
  progress: number;
  completed_samples: number;
  total_samples: number;
  recent_samples: RecentSample[];
}

interface GenerationProgressProps {
  experimentId: number;
  onComplete?: () => void;
}

export const GenerationProgress: React.FC<GenerationProgressProps> = ({
  experimentId,
  onComplete
}) => {
  const [variants, setVariants] = useState<VariantStatus[]>([]);
  const [isGenerating, setIsGenerating] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  useEffect(() => {
    let intervalId: number;

    const fetchProgress = async () => {
      try {
        // Add cache-busting timestamp to ensure fresh data
        const response = await fetch(
          `http://localhost:8000/experiments/${experimentId}/generation-status?t=${Date.now()}`
        );
        
        if (!response.ok) {
          throw new Error('Failed to fetch progress');
        }

        const data = await response.json();
        
        // Force state update with new array reference to ensure re-render
        setVariants([...data.variants]);

        // Check if all completed or cancelled
        if (data.all_completed || data.variants.some((v: VariantStatus) => v.status === 'cancelled')) {
          setIsGenerating(false);
          clearInterval(intervalId);
          
          // Call onComplete callback after a short delay
          setTimeout(() => {
            onComplete?.();
          }, 1000);
        }
      } catch (err) {
        console.error('Error fetching progress:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    // Initial fetch
    fetchProgress();

    // Poll every 1 second while generating for real-time updates
    if (isGenerating) {
      intervalId = setInterval(fetchProgress, 1000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [experimentId, isGenerating, onComplete]);

  const handleCancel = async () => {
    if (!confirm('Are you sure you want to cancel generation? Progress will be lost.')) {
      return;
    }
    
    setIsCancelling(true);
    try {
      await api.cancelGeneration(experimentId);
      toast.success('Generation cancelled');
      setIsGenerating(false);
      // Wait a moment for status update, then call onComplete
      setTimeout(() => {
        onComplete?.();
      }, 1000);
    } catch (err) {
      toast.error('Failed to cancel generation: ' + (err instanceof Error ? err.message : 'Unknown error'));
      setIsCancelling(false);
    }
  };

  const getStatusColor = (status: string): 'blue' | 'green' | 'yellow' | 'red' => {
    switch (status) {
      case 'completed':
        return 'green';
      case 'generating':
        return 'blue';
      case 'cancelled':
        return 'red';
      default:
        return 'yellow';
    }
  };

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-center space-x-2 text-red-800">
          <AlertCircle className="w-5 h-5" />
          <p>Error loading progress: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700">Generation Progress</h3>
        {isGenerating && !isCancelling && (
          <button
            onClick={handleCancel}
            className="flex items-center space-x-2 px-3 py-1.5 text-sm font-medium text-red-600 bg-red-50 hover:bg-red-100 rounded-md border border-red-200 transition-colors"
            disabled={isCancelling}
          >
            <X className="w-4 h-4" />
            <span>{isCancelling ? 'Cancelling...' : 'Cancel'}</span>
          </button>
        )}
      </div>
      <div className="space-y-3">
        {variants.map((variant) => (
          <div key={`${variant.variant_id}-${variant.progress}-${variant.completed_samples}`}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium text-gray-700">
                {variant.model_name}
              </span>
              <span className="text-xs text-gray-500">
                {variant.completed_samples}/{variant.total_samples}
              </span>
            </div>
            <ProgressBar
              progress={(variant.progress || 0) * 100}
              color={getStatusColor(variant.status)}
            />
            {variant.status === 'cancelled' && (
              <p className="text-xs text-red-600 mt-1">Cancelled</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};