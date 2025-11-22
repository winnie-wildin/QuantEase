//StatusBadge.tsx

import React from 'react';
import { CheckCircle, Clock, XCircle } from 'lucide-react';  // ← Removed AlertCircle

interface StatusBadgeProps {
  status: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  const statusConfig = {
    completed: {
      icon: CheckCircle,
      className: 'bg-emerald-100 text-emerald-700',
      label: 'Completed',
    },
    generating: {
      icon: Clock,
      className: 'bg-amber-100 text-amber-700',
      label: 'Generating...',
    },
    pending: {
      icon: Clock,
      className: 'bg-gray-100 text-gray-700',
      label: 'Pending',
    },
    failed: {
      icon: XCircle,
      className: 'bg-red-100 text-red-700',
      label: 'Failed',
    },
  };

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.pending;
  const Icon = config.icon;

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${config.className}`}>
      <Icon className="w-4 h-4 mr-1" />
      {config.label}
    </span>
  );
};