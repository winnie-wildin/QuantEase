import React, { useState } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { ArrowLeft, Upload, FileText, AlertCircle, CheckCircle2 } from 'lucide-react';
import toast from 'react-hot-toast';
import { api } from '../services/api';

interface ParsedSample {
  input_text: string;
  ground_truth_output?: string;
}

interface ValidationResult {
  isValid: boolean;
  samples: ParsedSample[];
  hasGroundTruth: boolean;
  sampleCount: number;
  errors: string[];
}

export const UploadDataset: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const experimentId = parseInt(id || '0');

  const [jsonText, setJsonText] = useState('');
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [uploading, setUploading] = useState(false);

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
        const output = item.output || item.ground_truth_output || item.ground_truth || item.answer || item.expected_output;
        
        const sample: ParsedSample = {
          input_text: input.trim()
        };

        if (output && typeof output === 'string' && output.trim()) {
          sample.ground_truth_output = output.trim();
          groundTruthCount++;
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
            // All samples have ground truth
            result.errors.push(`✅ All ${samples.length} samples have ground truth`);
          } else {
            // Partial ground truth
            result.errors.push(`⚠️ Only ${groundTruthCount}/${samples.length} samples have ground truth`);
          }
        } else {
          result.errors.push(`ℹ️ No ground truth detected - baseline will be required`);
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
      // Upload samples to backend
      const result = await api.uploadSamples(experimentId, validation.samples);

      toast.success(
        `✅ Uploaded ${validation.sampleCount} samples${result.has_ground_truth ? ' with ground truth' : ''}`, 
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

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Link to="/" className="text-gray-600 hover:text-gray-900 transition-colors">
          <ArrowLeft className="w-6 h-6" />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Upload Dataset</h1>
          <p className="text-gray-600">Upload your test samples as JSON</p>
        </div>
      </div>

      {/* Format Guide */}
      <div className="card bg-blue-50 border-blue-200">
        <h3 className="text-sm font-semibold text-blue-900 mb-2">📋 JSON Format Guide</h3>
        <div className="text-sm text-blue-800 space-y-2">
          <p><strong>Option 1: Inputs only</strong> (baseline model required)</p>
          <pre className="bg-white p-2 rounded text-xs overflow-x-auto">
{`[
  {"input": "What is AI?"},
  {"input": "Explain quantum computing"}
]`}
          </pre>
          <p><strong>Option 2: Inputs + Outputs</strong> (baseline optional)</p>
          <pre className="bg-white p-2 rounded text-xs overflow-x-auto">
{`[
  {
    "input": "What is AI?",
    "output": "AI is artificial intelligence..."
  },
  {
    "input": "Explain quantum computing",
    "output": "Quantum computing uses..."
  }
]`}
          </pre>
          <p className="text-xs text-blue-600">
            ℹ️ Flexible field names: <code>input/input_text/prompt</code> and <code>output/ground_truth_output/answer</code>
          </p>
        </div>
      </div>

      {/* Upload Area */}
      <div className="card">
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
                  <p>• Ground truth: <strong>{validation.hasGroundTruth ? 'Yes ✓' : 'No'}</strong></p>
                  {validation.hasGroundTruth && (
                    <p className="text-green-600">
                      ✨ You can skip baseline and compare directly against ground truth!
                    </p>
                  )}
                  {!validation.hasGroundTruth && (
                    <p className="text-yellow-700">
                      ⚠️ Baseline model will be required as reference
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
                <p className="text-gray-900 mt-1"><strong>Input:</strong> {sample.input_text.substring(0, 100)}{sample.input_text.length > 100 ? '...' : ''}</p>
                {sample.ground_truth_output && (
                  <p className="text-green-700 mt-1">
                    <strong>Output:</strong> {sample.ground_truth_output.substring(0, 100)}{sample.ground_truth_output.length > 100 ? '...' : ''}
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
          disabled={!validation || !validation.isValid || uploading}
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