//Home.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { FlaskConical, TrendingUp, Clock } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';

export const Home: React.FC = () => {
  const { data: experiments, isLoading } = useQuery({
    queryKey: ['experiments'],
    queryFn: api.getExperiments,
  });

  return (
    <div className="space-y-8">
      <div className="card">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Model Evaluation Dashboard
        </h2>
        <p className="text-gray-600 max-w-2xl">
          Compare baseline and quantized LLM models side-by-side. Track quality,
          speed, and resource trade-offs to make informed quantization decisions.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center space-x-4">
            <div className="p-3 rounded-lg bg-indigo-50">
  <FlaskConical className="w-6 h-6 text-indigo-600" />
</div>
            <div>
              <p className="text-sm text-gray-600">Total Experiments</p>
              <p className="text-2xl font-bold text-gray-900">
                {experiments?.length || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-green-100 rounded-lg">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Completed</p>
              <p className="text-2xl font-bold text-gray-900">
                {experiments?.filter(e => e.status === 'completed').length || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-amber-50 rounded-lg">
  <Clock className="w-6 h-6 text-amber-600" />
</div>
            <div>
              <p className="text-sm text-gray-600">In Progress</p>
              <p className="text-2xl font-bold text-gray-900">
                {experiments?.filter(e => e.status === 'generating').length || 0}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Experiments</h3>
        
        {isLoading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : experiments && experiments.length > 0 ? (
          <div className="space-y-3">
            {experiments.map(exp => (
              <Link
                key={exp.id}
                to={`/experiment/${exp.id}`}
                className="block p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold text-gray-900">{exp.name}</h4>
                    <p className="text-sm text-gray-600">
                      {exp.sample_count} samples • {exp.has_ground_truth ? 'With' : 'Without'} ground truth
                    </p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    exp.status === 'completed' ? 'bg-emerald-100 text-emerald-700' :
                    exp.status === 'generating' ? 'bg-amber-100 text-amber-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {exp.status}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            No experiments yet
          </div>
        )}
      </div>
    </div>
  );
};