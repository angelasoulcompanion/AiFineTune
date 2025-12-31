import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeft,
  CheckCircle,
  AlertCircle,
  Clock,
  Play,
  Trash2,
  RefreshCw,
} from 'lucide-react';
import { useDataset, useDatasetPreview, useValidateDataset, useDeleteDataset } from '../hooks/useDatasets';
import clsx from 'clsx';

const statusConfig = {
  pending: { icon: Clock, color: 'text-gray-500', label: 'Pending' },
  validating: { icon: RefreshCw, color: 'text-blue-500', label: 'Validating' },
  ready: { icon: CheckCircle, color: 'text-green-500', label: 'Ready' },
  error: { icon: AlertCircle, color: 'text-red-500', label: 'Error' },
  archived: { icon: Clock, color: 'text-gray-400', label: 'Archived' },
};

export default function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: dataset, isLoading, error } = useDataset(id!);
  const { data: preview, isLoading: previewLoading } = useDatasetPreview(id!, 10);
  const validateMutation = useValidateDataset();
  const deleteMutation = useDeleteDataset();

  const handleValidate = async () => {
    if (id) {
      await validateMutation.mutateAsync(id);
    }
  };

  const handleDelete = async () => {
    if (id && confirm('Are you sure you want to delete this dataset?')) {
      await deleteMutation.mutateAsync(id);
      navigate('/datasets');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-600 dark:text-red-400">Dataset not found</p>
        <Link to="/datasets" className="text-primary-600 hover:underline mt-2 inline-block">
          Back to datasets
        </Link>
      </div>
    );
  }

  const status = statusConfig[dataset.status];
  const StatusIcon = status.icon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            to="/datasets"
            className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-700 mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to datasets
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {dataset.name}
          </h1>
          {dataset.description && (
            <p className="text-gray-600 dark:text-gray-400 mt-1">{dataset.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {dataset.status === 'pending' && (
            <button
              onClick={handleValidate}
              disabled={validateMutation.isPending}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
            >
              {validateMutation.isPending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Validate
            </button>
          )}
          {dataset.status === 'ready' && (
            <Link
              to={`/training/new?dataset=${dataset.dataset_id}`}
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg"
            >
              <Play className="w-4 h-4" />
              Start Training
            </Link>
          )}
          <button
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
            className="p-2 text-gray-400 hover:text-red-500 transition-colors"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Status and Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Status</p>
          <div className="flex items-center gap-2 mt-1">
            <StatusIcon className={clsx('w-5 h-5', status.color)} />
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {status.label}
            </span>
          </div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Type</p>
          <p className="font-medium text-gray-900 dark:text-gray-100 mt-1">
            {dataset.dataset_type.toUpperCase()}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Examples</p>
          <p className="font-medium text-gray-900 dark:text-gray-100 mt-1">
            {dataset.total_examples?.toLocaleString() || '-'}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Size</p>
          <p className="font-medium text-gray-900 dark:text-gray-100 mt-1">
            {dataset.file_size_bytes
              ? `${(dataset.file_size_bytes / (1024 * 1024)).toFixed(2)} MB`
              : '-'}
          </p>
        </div>
      </div>

      {/* Validation Errors */}
      {dataset.validation_errors && dataset.validation_errors.length > 0 && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <h3 className="font-medium text-red-600 dark:text-red-400 mb-2">
            Validation Errors ({dataset.validation_errors.length})
          </h3>
          <ul className="space-y-1 text-sm text-red-600 dark:text-red-400">
            {dataset.validation_errors.slice(0, 5).map((err, i) => (
              <li key={i}>
                Row {err.row}: {err.message}
              </li>
            ))}
            {dataset.validation_errors.length > 5 && (
              <li>... and {dataset.validation_errors.length - 5} more</li>
            )}
          </ul>
        </div>
      )}

      {/* Validation Warnings */}
      {dataset.validation_warnings && dataset.validation_warnings.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <h3 className="font-medium text-yellow-600 dark:text-yellow-400 mb-2">
            Warnings ({dataset.validation_warnings.length})
          </h3>
          <ul className="space-y-1 text-sm text-yellow-600 dark:text-yellow-400">
            {dataset.validation_warnings.slice(0, 5).map((warn, i) => (
              <li key={i}>
                {warn.column}: {warn.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Statistics */}
      {dataset.is_validated && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Statistics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Train Examples</p>
              <p className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {dataset.train_examples?.toLocaleString() || '-'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Validation Examples</p>
              <p className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {dataset.validation_examples?.toLocaleString() || '-'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Avg Input Length</p>
              <p className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {dataset.avg_input_length?.toLocaleString() || '-'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Avg Output Length</p>
              <p className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                {dataset.avg_output_length?.toLocaleString() || '-'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Preview */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Data Preview
        </h3>
        {previewLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500"></div>
          </div>
        ) : preview ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 text-sm">
              <thead>
                <tr>
                  {preview.columns.map((col) => (
                    <th
                      key={col}
                      className="px-4 py-2 text-left font-medium text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-700/50"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {preview.rows.map((row, i) => (
                  <tr key={i}>
                    {preview.columns.map((col) => (
                      <td
                        key={col}
                        className="px-4 py-2 text-gray-600 dark:text-gray-300 max-w-xs truncate"
                      >
                        {typeof row[col] === 'object'
                          ? JSON.stringify(row[col]).slice(0, 100)
                          : String(row[col] ?? '').slice(0, 100)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Showing {preview.preview_rows} of {preview.total_rows} rows
            </p>
          </div>
        ) : (
          <p className="text-gray-500 dark:text-gray-400">No preview available</p>
        )}
      </div>
    </div>
  );
}
