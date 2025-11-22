import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { ArrowLeft, FlaskConical } from 'lucide-react';
import { api } from '../services/api';

export const CreateExperiment: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    baseline_model_id: 1,
    has_ground_truth: false,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const experiment = await api.createExperiment(formData);
      navigate(`/experiment/${experiment.id}/upload`);
    } catch (error) {
      console.error('Failed to create experiment:', error);
      alert('Failed to create experiment');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex items-center space-x-4">
        <Link to="/" className="text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-6 h-6" />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Create Experiment</h1>
          <p className="text-gray-600">Set up a new comparison experiment</p>
        </div>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Experiment Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Experiment Name *
            </label>
            <input
              type="text"
              required
              className="input"
              placeholder="e.g., Llama-2 7B Quantization Test"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            />
            <p className="text-sm text-gray-500 mt-1">
              Give your experiment a descriptive name
            </p>
          </div>


          {/* Info Box */}
          <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <FlaskConical className="w-5 h-5 text-primary-600 mt-0.5" />
              <div className="text-sm text-gray-700">
                <p className="font-medium mb-1">What happens next?</p>
                <ul className="list-disc list-inside space-y-1 text-gray-600">
                  <li>Upload your dataset (CSV or JSON)</li>
                  <li>Select models to compare (baseline + quantized)</li>
                  <li>Run generation and evaluation</li>
                  <li>View side-by-side comparison results</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Buttons */}
          <div className="flex justify-end space-x-4">
            <Link to="/" className="btn-secondary">
              Cancel
            </Link>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary"
            >
              {loading ? 'Creating...' : 'Create & Continue →'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};