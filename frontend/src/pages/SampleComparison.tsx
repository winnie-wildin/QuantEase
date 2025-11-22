import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, ChevronLeft, ChevronRight, Eye } from 'lucide-react';

interface QuantizedOutput {
  variant_id: number;
  model_name: string;
  output_text: string;
  latency_ms: number;
  similarity_score: number | null;
}

interface ComparisonSample {
  sample_id: number;
  position: number;
  input_text: string;
  ground_truth?: string;
  baseline_output?: string;
  baseline_latency?: number;
  quantized_outputs: QuantizedOutput[];
}

interface ComparisonData {
  experiment_id: number;
  has_ground_truth: boolean;
  has_baseline: boolean;
  quantized_count: number;
  total_samples: number;
  page: number;
  page_size: number;
  total_pages: number;
  samples: ComparisonSample[];
}

export const SampleComparison: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const experimentId = parseInt(id || '0');
  const [page, setPage] = useState(1);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  const { data, isLoading } = useQuery<ComparisonData>({
    queryKey: ['sample-comparison', experimentId, page],
    queryFn: async () => {
      const response = await fetch(
        `http://localhost:8000/experiments/${experimentId}/samples/comparison?page=${page}&page_size=20`
      );
      if (!response.ok) throw new Error('Failed to fetch comparison data');
      return response.json();
    },
  });

  const getScoreColor = (score?: number | null) => {
    if (!score) return 'text-gray-400';
    if (score >= 0.6) return 'text-green-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBg = (score?: number | null) => {
    if (!score) return 'bg-gray-100';
    if (score >= 0.6) return 'bg-green-50';
    if (score >= 0.4) return 'bg-yellow-50';
    return 'bg-red-50';
  };

  const truncate = (text: string, maxLength: number = 100) => {
    if (!text) return '—';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-primary-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading samples...</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return <div className="text-center py-12 text-red-600">Failed to load comparison data</div>;
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Link to={`/experiment/${experimentId}`} className="text-gray-600 hover:text-gray-900">
          <ArrowLeft className="w-6 h-6" />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Sample Outputs</h1>
          <p className="text-gray-600">
            {data.total_samples} samples compared •
            {data.has_ground_truth && ' Ground truth available •'}
            {data.has_baseline && ' Baseline included •'}
            {` ${data.quantized_count} quantized model${data.quantized_count !== 1 ? 's' : ''}`}
          </p>
        </div>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase w-12">#</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Input</th>
                
                {/* Ground Truth Column (if exists) */}
                {data.has_ground_truth && (
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Ground Truth
                  </th>
                )}
                
                {/* Baseline Column (if exists) */}
                {data.has_baseline && (
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Baseline Output
                  </th>
                )}
                
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Quantized Output
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase w-24">
                  Similarity
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase w-16">
                  Action
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {data.samples.map((sample) => {
                const quantOutput = sample.quantized_outputs[0]; // Get first quantized output
                
                return (
                  <React.Fragment key={sample.sample_id}>
                    <tr className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm text-gray-500 font-medium">
                        {sample.position + 1}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900 max-w-xs">
                        {truncate(sample.input_text, 80)}
                      </td>
                      
                      {/* Ground Truth */}
                      {data.has_ground_truth && (
                        <td className="px-4 py-3 text-sm text-green-700 max-w-xs">
                          {truncate(sample.ground_truth || '', 80)}
                        </td>
                      )}
                      
                      {/* Baseline Output */}
                      {data.has_baseline && (
                        <td className="px-4 py-3 text-sm text-gray-700 max-w-xs">
                          {truncate(sample.baseline_output || '', 80)}
                          {sample.baseline_latency && (
                            <span className="text-xs text-gray-500 block mt-1">
                              ⏱️ {sample.baseline_latency.toFixed(0)}ms
                            </span>
                          )}
                        </td>
                      )}
                      
                      {/* Quantized Output */}
                      <td className="px-4 py-3 text-sm text-gray-700 max-w-xs">
                        {quantOutput ? (
                          <>
                            {truncate(quantOutput.output_text, 80)}
                            <span className="text-xs text-gray-500 block mt-1">
                              ⏱️ {quantOutput.latency_ms.toFixed(0)}ms
                            </span>
                          </>
                        ) : (
                          '—'
                        )}
                      </td>
                      
                      {/* Similarity Score */}
                      <td className={`px-4 py-3 text-center ${getScoreBg(quantOutput?.similarity_score)}`}>
                        <span className={`font-bold text-sm ${getScoreColor(quantOutput?.similarity_score)}`}>
                          {quantOutput?.similarity_score 
                            ? `${(quantOutput.similarity_score * 100).toFixed(1)}%` 
                            : '—'
                          }
                        </span>
                      </td>
                      
                      {/* Action */}
                      <td className="px-4 py-3 text-center">
                        <button
                          onClick={() => setExpandedRow(expandedRow === sample.sample_id ? null : sample.sample_id)}
                          className="text-primary-600 hover:text-primary-700 p-1 hover:bg-primary-50 rounded"
                          title="View full details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                    
                    {/* Expanded Row */}
                    {expandedRow === sample.sample_id && (
                      <tr className="bg-gray-50">
                        <td colSpan={7} className="px-4 py-6">
                          <div className="space-y-4 max-w-5xl">
                            {/* Input */}
                            <div className="bg-white p-4 rounded-lg border border-gray-200">
                              <p className="text-xs font-semibold text-gray-500 uppercase mb-2">Input:</p>
                              <p className="text-sm text-gray-900">{sample.input_text}</p>
                            </div>
                            
                            {/* Ground Truth */}
                            {sample.ground_truth && (
                              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                <p className="text-xs font-semibold text-green-700 uppercase mb-2">Ground Truth:</p>
                                <p className="text-sm text-gray-900">{sample.ground_truth}</p>
                              </div>
                            )}
                            
                            {/* Outputs Side by Side */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {/* Baseline */}
                              {data.has_baseline && (
                                <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                  <p className="text-xs font-semibold text-blue-700 uppercase mb-2">
                                    Baseline Output:
                                  </p>
                                  <p className="text-sm text-gray-900 mb-2">
                                    {sample.baseline_output || '—'}
                                  </p>
                                  {sample.baseline_latency && (
                                    <p className="text-xs text-gray-600">
                                      ⏱️ Latency: {sample.baseline_latency.toFixed(0)}ms
                                    </p>
                                  )}
                                </div>
                              )}
                              
                              {/* Quantized */}
                              {quantOutput && (
                                <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                                  <p className="text-xs font-semibold text-purple-700 uppercase mb-2">
                                    Quantized Output ({quantOutput.model_name}):
                                  </p>
                                  <p className="text-sm text-gray-900 mb-2">
                                    {quantOutput.output_text}
                                  </p>
                                  <div className="flex items-center justify-between text-xs text-gray-600">
                                    <span>⏱️ Latency: {quantOutput.latency_ms.toFixed(0)}ms</span>
                                    {quantOutput.similarity_score && (
                                      <span className={getScoreColor(quantOutput.similarity_score)}>
                                        Similarity: {(quantOutput.similarity_score * 100).toFixed(1)}%
                                      </span>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-200 flex items-center justify-between">
          <p className="text-sm text-gray-700">
            Showing {((page - 1) * 20) + 1} to {Math.min(page * 20, data.total_samples)} of {data.total_samples} samples
          </p>
          <div className="flex space-x-2">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-1"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Previous</span>
            </button>
            <button
              onClick={() => setPage(p => p + 1)}
              disabled={page >= data.total_pages}
              className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-1"
            >
              <span>Next</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};