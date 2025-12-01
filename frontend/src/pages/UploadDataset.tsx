// frontend/src/pages/UploadDataset.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { ArrowLeft, Upload, FileText, AlertCircle, CheckCircle2, Zap } from 'lucide-react';
import toast from 'react-hot-toast';
import { api } from '../services/api';

interface ParsedSample {
  input_text: string;
  ground_truth_output?: string;
  context?: string;
}

interface ValidationResult {
  isValid: boolean;
  samples: ParsedSample[];
  hasGroundTruth: boolean;
  sampleCount: number;
  errors: string[];
}

type TaskType = 'text_generation' | 'classification' | 'rag';

export const UploadDataset: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const experimentId = parseInt(id || '0');

  const [taskType, setTaskType] = useState<TaskType>('text_generation');
  const [jsonText, setJsonText] = useState('');
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [recommendedModel, setRecommendedModel] = useState<any>(null);

  // Fetch recommended model when task type changes
  useEffect(() => {
    const fetchRecommendation = async () => {
      try {
        const response = await fetch(`http://localhost:8000/experiments/models/recommendations/${taskType}`);
        const data = await response.json();
        setRecommendedModel(data);
      } catch (error) {
        console.error('Failed to fetch recommendation:', error);
      }
    };
    fetchRecommendation();
  }, [taskType]);

  // Validate and parse JSON
  const validateJSON = (text: string): ValidationResult => {
    const result: ValidationResult = {
      isValid: false,
      samples: [],
      hasGroundTruth: false,
      sampleCount: 0,
      errors: []
    };

    if (!text.trim()) {
      result.errors.push('JSON is empty');
      return result;
    }

    try {
      const parsed = JSON.parse(text);

      // Check if it's an array
      if (!Array.isArray(parsed)) {
        result.errors.push('JSON must be an array of samples');
        return result;
      }

      if (parsed.length === 0) {
        result.errors.push('Array is empty - need at least 1 sample');
        return result;
      }

      // Parse each sample
      let groundTruthCount = 0;
      const samples: ParsedSample[] = [];

      parsed.forEach((item, index) => {
        // Try to extract input
        const input = item.input || item.input_text || item.prompt || item.question || item.text;
        
        if (!input || typeof input !== 'string') {
          result.errors.push(`Sample ${index + 1}: Missing or invalid 'input' field`);
          return;
        }

        // Try to extract output (optional)
        const output = item.output || item.ground_truth_output || item.ground_truth || item.answer || item.expected_output || item.label;
        
        // For RAG: extract context
        const context = item.context || item.retrieved_context || item.document;

        const sample: ParsedSample = {
          input_text: input.trim()
        };

        if (output && typeof output === 'string' && output.trim()) {
          sample.ground_truth_output = output.trim();
          groundTruthCount++;
        }

        if (context && typeof context === 'string') {
          sample.context = context.trim();
        }

        samples.push(sample);
      });

      // If any samples were parsed
      if (samples.length > 0) {
        result.samples = samples;
        result.sampleCount = samples.length;
        result.hasGroundTruth = groundTruthCount > 0;
        result.isValid = result.errors.length === 0;

        // Info messages
        if (result.hasGroundTruth) {
          if (groundTruthCount === samples.length) {
            result.errors.push(`✅ All ${samples.length} samples have ground truth`);
          } else {
            result.errors.push(`⚠️ Only ${groundTruthCount}/${samples.length} samples have ground truth`);
          }
        } else {
          result.errors.push(`ℹ️ No ground truth detected - baseline will be required`);
        }

        // RAG-specific validation
        if (taskType === 'rag') {
          const samplesWithContext = samples.filter(s => s.context).length;
          if (samplesWithContext > 0) {
            result.errors.push(`📚 Found context in ${samplesWithContext}/${samples.length} samples`);
          }
        }
      }

    } catch (error) {
      result.errors.push(`Invalid JSON: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }

    return result;
  };

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.json')) {
      toast.error('Please upload a JSON file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      setJsonText(text);
      const result = validateJSON(text);
      setValidation(result);
    };
    reader.onerror = () => {
      toast.error('Failed to read file');
    };
    reader.readAsText(file);
  };

  // Handle text change
  const handleTextChange = (text: string) => {
    setJsonText(text);
    if (text.trim()) {
      const result = validateJSON(text);
      setValidation(result);
    } else {
      setValidation(null);
    }
  };

  // Submit samples
  const handleSubmit = async () => {
    if (!validation || !validation.isValid) {
      toast.error('Please fix validation errors first');
      return;
    }

    setUploading(true);
    const toastId = toast.loading('Uploading samples...');

    try {
      // Upload samples with task type
      const result = await api.uploadSamples(experimentId, validation.samples, taskType);

      toast.success(
        `✅ Uploaded ${validation.sampleCount} samples for ${taskType.replace('_', ' ')}`, 
        { id: toastId }
      );
      
      // Navigate to model selection
      navigate(`/experiment/${experimentId}/setup`);
    } catch (error) {
      console.error('Upload failed:', error);
      toast.error('Failed to upload samples: ' + error, { id: toastId });
    } finally {
      setUploading(false);
    }
  };

  // Get example JSON for task type
  const getExampleJSON = () => {
    if (taskType === 'text_generation') {
      return `[
  {
    "input": "What is AI?",
    "output": "AI is artificial intelligence..."
  },
  {
    "input": "Explain quantum computing",
    "output": "Quantum computing uses..."
  }
]`;
    } else if (taskType === 'classification') {
      return `[
  {
    "input": "I love this product!",
    "label": "positive"
  },
  {
    "input": "This is terrible",
    "label": "negative"
  }
]`;
    } else {
      return `[
  {
    "input": "What is the capital of France?",
    "context": "France is a country in Europe...",
    "output": "Paris"
  }
]`;
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Link to="/" className="text-gray-600 hover:text-gray-900 transition-colors">
          <ArrowLeft className="w-6 h-6" />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Upload Dataset</h1>
          <p className="text-gray-600">Choose task type and upload your test samples</p>
        </div>
      </div>

      {/* Task Type Selection */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">1. Select Task Type</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Text Generation */}
          <label className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
            taskType === 'text_generation' 
              ? 'border-primary-500 bg-primary-50 shadow-sm' 
              : 'border-gray-200 hover:border-gray-300'
          }`}>
            <input
              type="radio"
              name="taskType"
              value="text_generation"
              checked={taskType === 'text_generation'}
              onChange={(e) => setTaskType(e.target.value as TaskType)}
              className="sr-only"
            />
            <div className="flex items-start space-x-3">
              {taskType === 'text_generation' && (
                <CheckCircle2 className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <p className="font-semibold text-gray-900">🔤 Text Generation</p>
                <p className="text-sm text-gray-600 mt-1">
                  For creative writing, translations, Q&A, summaries
                </p>
              </div>
            </div>
          </label>

          {/* Classification */}
          <label className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
            taskType === 'classification' 
              ? 'border-primary-500 bg-primary-50 shadow-sm' 
              : 'border-gray-200 hover:border-gray-300'
          }`}>
            <input
              type="radio"
              name="taskType"
              value="classification"
              checked={taskType === 'classification'}
              onChange={(e) => setTaskType(e.target.value as TaskType)}
              className="sr-only"
            />
            <div className="flex items-start space-x-3">
              {taskType === 'classification' && (
                <CheckCircle2 className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <p className="font-semibold text-gray-900">🎯 Classification</p>
                <p className="text-sm text-gray-600 mt-1">
                  For sentiment analysis, categorization, labeling
                </p>
              </div>
            </div>
          </label>

          {/* RAG */}
          <label className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
            taskType === 'rag' 
              ? 'border-primary-500 bg-primary-50 shadow-sm' 
              : 'border-gray-200 hover:border-gray-300'
          }`}>
            <input
              type="radio"
              name="taskType"
              value="rag"
              checked={taskType === 'rag'}
              onChange={(e) => setTaskType(e.target.value as TaskType)}
              className="sr-only"
            />
            <div className="flex items-start space-x-3">
              {taskType === 'rag' && (
                <CheckCircle2 className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <p className="font-semibold text-gray-900">🔍 RAG / Q&A</p>
                <p className="text-sm text-gray-600 mt-1">
                  For retrieval-based question answering
                </p>
              </div>
            </div>
          </label>
        </div>

        {/* Recommended Model */}
        {recommendedModel && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-blue-600" />
              <p className="text-sm text-blue-800">
                <strong>Recommended baseline:</strong> {recommendedModel.display_name}
              </p>
            </div>
            <p className="text-xs text-blue-600 mt-1 ml-6">{recommendedModel.reason}</p>
          </div>
        )}
      </div>

      {/* Format Guide */}
      <div className="card bg-blue-50 border-blue-200">
        <h3 className="text-sm font-semibold text-blue-900 mb-2">
          2. JSON Format for {taskType.replace('_', ' ')}
        </h3>
        <div className="text-sm text-blue-800 space-y-2">
          <pre className="bg-white p-3 rounded text-xs overflow-x-auto">
            {getExampleJSON()}
          </pre>
          <p className="text-xs text-blue-600">
            {taskType === 'text_generation' && 'ℹ️ Flexible field names: input/prompt and output/ground_truth'}
            {taskType === 'classification' && 'ℹ️ Use "label" for ground truth class (required for classification)'}
            {taskType === 'rag' && 'ℹ️ Include "context" field with retrieved documents/passages'}
          </p>
        </div>
      </div>

      {/* Upload Area */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">3. Upload Your Data</h3>
        <div className="space-y-4">
          {/* File Upload Button */}
          <div>
            <label className="btn-primary cursor-pointer inline-flex items-center space-x-2">
              <Upload className="w-4 h-4" />
              <span>Upload JSON File</span>
              <input
                type="file"
                accept=".json"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
            <p className="text-sm text-gray-500 mt-2">Or paste JSON below</p>
          </div>

          {/* Text Area */}
          <div>
            <textarea
              value={jsonText}
              onChange={(e) => handleTextChange(e.target.value)}
              placeholder='Paste your JSON here...'
              className="w-full h-64 p-4 border-2 border-gray-300 rounded-lg font-mono text-sm focus:border-primary-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* Validation Results */}
      {validation && (
        <div className={`card ${validation.isValid ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div className="flex items-start space-x-3">
            {validation.isValid ? (
              <CheckCircle2 className="w-6 h-6 text-green-600 flex-shrink-0" />
            ) : (
              <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0" />
            )}
            <div className="flex-1">
              <h3 className={`font-semibold mb-2 ${validation.isValid ? 'text-green-900' : 'text-red-900'}`}>
                {validation.isValid ? 'Validation Passed ✓' : 'Validation Failed ✗'}
              </h3>
              
              {validation.isValid && (
                <div className="text-sm text-green-800 space-y-1">
                  <p>• Found <strong>{validation.sampleCount} samples</strong></p>
                  <p>• Task type: <strong>{taskType.replace('_', ' ')}</strong></p>
                  <p>• Ground truth: <strong>{validation.hasGroundTruth ? 'Yes ✓' : 'No'}</strong></p>
                  {taskType === 'classification' && !validation.hasGroundTruth && (
                    <p className="text-red-600 font-medium">
                      ⚠️ Classification requires ground truth labels!
                    </p>
                  )}
                </div>
              )}

              {validation.errors.length > 0 && (
                <ul className={`text-sm space-y-1 mt-2 ${validation.isValid ? 'text-green-700' : 'text-red-700'}`}>
                  {validation.errors.map((error, i) => (
                    <li key={i}>• {error}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Sample Preview */}
      {validation && validation.isValid && validation.samples.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-3">📝 Sample Preview (first 3)</h3>
          <div className="space-y-3">
            {validation.samples.slice(0, 3).map((sample, i) => (
              <div key={i} className="border border-gray-200 rounded p-3 text-sm">
                <p className="text-gray-600 font-medium">Sample {i + 1}:</p>
                <p className="text-gray-900 mt-1">
                  <strong>Input:</strong> {sample.input_text.substring(0, 100)}
                  {sample.input_text.length > 100 ? '...' : ''}
                </p>
                {sample.context && (
                  <p className="text-purple-700 mt-1">
                    <strong>Context:</strong> {sample.context.substring(0, 100)}
                    {sample.context.length > 100 ? '...' : ''}
                  </p>
                )}
                {sample.ground_truth_output && (
                  <p className="text-green-700 mt-1">
                    <strong>{taskType === 'classification' ? 'Label' : 'Output'}:</strong> {sample.ground_truth_output.substring(0, 100)}
                    {sample.ground_truth_output.length > 100 ? '...' : ''}
                  </p>
                )}
              </div>
            ))}
            {validation.samples.length > 3 && (
              <p className="text-sm text-gray-500 text-center">
                ... and {validation.samples.length - 3} more samples
              </p>
            )}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-between items-center">
        <Link to="/" className="btn-secondary">
          Cancel
        </Link>
        <button
          onClick={handleSubmit}
          disabled={!validation || !validation.isValid || uploading || (taskType === 'classification' && !validation.hasGroundTruth)}
          className="btn-primary flex items-center space-x-2"
        >
          {uploading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Uploading...</span>
            </>
          ) : (
            <>
              <FileText className="w-4 h-4" />
              <span>Continue to Model Selection</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};