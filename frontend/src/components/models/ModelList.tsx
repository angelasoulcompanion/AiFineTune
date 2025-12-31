import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Box,
  Plus,
  Search,
  CheckCircle,
  AlertCircle,
  Download,
  Trash2,
  ExternalLink,
} from 'lucide-react';
import { useModels, useDeleteModel } from '../../hooks/useModels';
import type { Model } from '../../hooks/useModels';
import clsx from 'clsx';

const statusConfig = {
  available: { icon: CheckCircle, color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-700', label: 'Available' },
  downloading: { icon: Download, color: 'text-blue-500', bg: 'bg-blue-100 dark:bg-blue-900/30', label: 'Downloading' },
  ready: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100 dark:bg-green-900/30', label: 'Ready' },
  error: { icon: AlertCircle, color: 'text-red-500', bg: 'bg-red-100 dark:bg-red-900/30', label: 'Error' },
};

const typeConfig = {
  base: { label: 'Base', color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300' },
  lora: { label: 'LoRA', color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' },
  merged: { label: 'Merged', color: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' },
};

interface ModelRowProps {
  model: Model;
  onDelete: (id: string) => void;
}

function ModelRow({ model, onDelete }: ModelRowProps) {
  const status = statusConfig[model.status];
  const StatusIcon = status.icon;
  const type = typeConfig[model.model_type];

  return (
    <tr className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
      <td className="px-6 py-4">
        <Link
          to={`/models/${model.model_id}`}
          className="font-medium text-gray-900 dark:text-gray-100 hover:text-primary-600"
        >
          {model.name}
        </Link>
        {model.description && (
          <p className="text-sm text-gray-500 dark:text-gray-400 truncate max-w-xs">
            {model.description}
          </p>
        )}
      </td>
      <td className="px-6 py-4">
        <span className={clsx('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', type.color)}>
          {type.label}
        </span>
      </td>
      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
        <span className="font-mono text-xs">{model.base_model_id}</span>
      </td>
      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
        {model.file_size_mb ? `${model.file_size_mb.toFixed(1)} MB` : '-'}
      </td>
      <td className="px-6 py-4">
        <span className={clsx('inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs', status.bg)}>
          <StatusIcon className={clsx('w-3 h-3', status.color)} />
          <span className={status.color}>{status.label}</span>
        </span>
      </td>
      <td className="px-6 py-4">
        <div className="flex items-center gap-2">
          {model.is_pushed_to_hf && (
            <a
              href={`https://huggingface.co/${model.hf_repo_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-primary-500"
              title="View on HuggingFace"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          )}
          <button
            onClick={() => onDelete(model.model_id)}
            className="text-gray-400 hover:text-red-500 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </td>
    </tr>
  );
}

export default function ModelList() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const { data, isLoading, error } = useModels({ page, per_page: 20 });
  const deleteMutation = useDeleteModel();

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this model?')) {
      await deleteMutation.mutateAsync(id);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-600 dark:text-red-400">
        Failed to load models
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Models</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Manage your base models and LoRA adapters
          </p>
        </div>
        <Link
          to="/models/new"
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Model
        </Link>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search models..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
        />
      </div>

      {/* Table */}
      {data?.models.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
          <Box className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No models yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Add a base model from HuggingFace to start fine-tuning
          </p>
          <Link
            to="/models/new"
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            Add Model
          </Link>
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Base Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Size
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {data?.models.map((model) => (
                <ModelRow
                  key={model.model_id}
                  model={model}
                  onDelete={handleDelete}
                />
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          {data && data.total > data.per_page && (
            <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Showing {(page - 1) * data.per_page + 1} to {Math.min(page * data.per_page, data.total)} of {data.total}
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded text-sm disabled:opacity-50"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage((p) => p + 1)}
                  disabled={page * data.per_page >= data.total}
                  className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded text-sm disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
