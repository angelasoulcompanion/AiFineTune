import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Database,
  Plus,
  Search,
  CheckCircle,
  AlertCircle,
  Clock,
  FileText,
  Trash2,
} from 'lucide-react';
import { useDatasets, useDeleteDataset } from '../../hooks/useDatasets';
import type { Dataset } from '../../hooks/useDatasets';
import clsx from 'clsx';

const statusConfig = {
  pending: { icon: Clock, color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-700' },
  validating: { icon: Clock, color: 'text-blue-500', bg: 'bg-blue-100 dark:bg-blue-900/30' },
  ready: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100 dark:bg-green-900/30' },
  error: { icon: AlertCircle, color: 'text-red-500', bg: 'bg-red-100 dark:bg-red-900/30' },
  archived: { icon: FileText, color: 'text-gray-400', bg: 'bg-gray-100 dark:bg-gray-700' },
};

const formatBytes = (bytes: number | null): string => {
  if (!bytes) return '-';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

interface DatasetRowProps {
  dataset: Dataset;
  onDelete: (id: string) => void;
}

function DatasetRow({ dataset, onDelete }: DatasetRowProps) {
  const status = statusConfig[dataset.status];
  const StatusIcon = status.icon;

  return (
    <tr className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
      <td className="px-6 py-4">
        <Link
          to={`/datasets/${dataset.dataset_id}`}
          className="font-medium text-gray-900 dark:text-gray-100 hover:text-primary-600"
        >
          {dataset.name}
        </Link>
        {dataset.description && (
          <p className="text-sm text-gray-500 dark:text-gray-400 truncate max-w-xs">
            {dataset.description}
          </p>
        )}
      </td>
      <td className="px-6 py-4">
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
          {dataset.dataset_type.toUpperCase()}
        </span>
      </td>
      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
        {dataset.format.toUpperCase()}
      </td>
      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
        {dataset.total_examples?.toLocaleString() || '-'}
      </td>
      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
        {formatBytes(dataset.file_size_bytes)}
      </td>
      <td className="px-6 py-4">
        <span className={clsx('inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs', status.bg)}>
          <StatusIcon className={clsx('w-3 h-3', status.color)} />
          <span className={status.color}>{dataset.status}</span>
        </span>
      </td>
      <td className="px-6 py-4 text-right">
        <button
          onClick={() => onDelete(dataset.dataset_id)}
          className="text-gray-400 hover:text-red-500 transition-colors"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </td>
    </tr>
  );
}

export default function DatasetList() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const { data, isLoading, error } = useDatasets({ page, per_page: 20 });
  const deleteMutation = useDeleteDataset();

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this dataset?')) {
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
        Failed to load datasets
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Datasets</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Manage your training datasets
          </p>
        </div>
        <Link
          to="/datasets/new"
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          Upload Dataset
        </Link>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search datasets..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
        />
      </div>

      {/* Table */}
      {data?.datasets.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
          <Database className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No datasets yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Upload your first dataset to start fine-tuning
          </p>
          <Link
            to="/datasets/new"
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            Upload Dataset
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
                  Format
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Examples
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
              {data?.datasets.map((dataset) => (
                <DatasetRow
                  key={dataset.dataset_id}
                  dataset={dataset}
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
