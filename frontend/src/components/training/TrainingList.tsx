import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Play,
  Plus,
  CheckCircle,
  AlertCircle,
  Clock,
  Loader2,
  Trash2,
  StopCircle,
  XCircle,
} from 'lucide-react';
import {
  useTrainingJobs,
  useDeleteTrainingJob,
  useCancelTrainingJob,
} from '../../hooks/useTraining';
import type { TrainingJob } from '../../hooks/useTraining';
import clsx from 'clsx';

const statusConfig = {
  queued: { icon: Clock, color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-700', label: 'Queued', animate: false },
  preparing: { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-100 dark:bg-blue-900/30', label: 'Preparing', animate: true },
  training: { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-100 dark:bg-blue-900/30', label: 'Training', animate: true },
  evaluating: { icon: Loader2, color: 'text-purple-500', bg: 'bg-purple-100 dark:bg-purple-900/30', label: 'Evaluating', animate: true },
  saving: { icon: Loader2, color: 'text-yellow-500', bg: 'bg-yellow-100 dark:bg-yellow-900/30', label: 'Saving', animate: true },
  completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100 dark:bg-green-900/30', label: 'Completed', animate: false },
  failed: { icon: AlertCircle, color: 'text-red-500', bg: 'bg-red-100 dark:bg-red-900/30', label: 'Failed', animate: false },
  cancelled: { icon: XCircle, color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-700', label: 'Cancelled', animate: false },
};

const methodConfig = {
  sft: { label: 'SFT', color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300' },
  lora: { label: 'LoRA', color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' },
  qlora: { label: 'QLoRA', color: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300' },
  dpo: { label: 'DPO', color: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300' },
  orpo: { label: 'ORPO', color: 'bg-pink-100 text-pink-800 dark:bg-pink-900/30 dark:text-pink-300' },
};

interface JobRowProps {
  job: TrainingJob;
  onDelete: (id: string) => void;
  onCancel: (id: string) => void;
}

function JobRow({ job, onDelete, onCancel }: JobRowProps) {
  const status = statusConfig[job.status];
  const StatusIcon = status.icon;
  const method = methodConfig[job.training_method];
  const isRunning = ['preparing', 'training', 'evaluating', 'saving'].includes(job.status);

  return (
    <tr className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
      <td className="px-6 py-4">
        <Link
          to={`/training/${job.job_id}`}
          className="font-medium text-gray-900 dark:text-gray-100 hover:text-primary-600"
        >
          {job.name}
        </Link>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {job.dataset_name} → {job.model_name}
        </div>
      </td>
      <td className="px-6 py-4">
        <span className={clsx('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', method.color)}>
          {method.label}
        </span>
      </td>
      <td className="px-6 py-4">
        <div className="flex items-center gap-2">
          <span className={clsx('inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs', status.bg)}>
            <StatusIcon className={clsx('w-3 h-3', status.color, status.animate && 'animate-spin')} />
            <span className={status.color}>{status.label}</span>
          </span>
        </div>
        {isRunning && job.progress_percentage !== undefined && (
          <div className="mt-1 w-24">
            <div className="h-1.5 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 transition-all duration-300"
                style={{ width: `${job.progress_percentage}%` }}
              />
            </div>
            <span className="text-xs text-gray-500">{job.progress_percentage.toFixed(1)}%</span>
          </div>
        )}
      </td>
      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
        {job.current_loss !== undefined ? job.current_loss.toFixed(4) : '-'}
      </td>
      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
        {new Date(job.created_at).toLocaleDateString()}
      </td>
      <td className="px-6 py-4">
        <div className="flex items-center gap-2">
          {isRunning && (
            <button
              onClick={() => onCancel(job.job_id)}
              className="text-gray-400 hover:text-orange-500 transition-colors"
              title="Cancel"
            >
              <StopCircle className="w-4 h-4" />
            </button>
          )}
          {!isRunning && (
            <button
              onClick={() => onDelete(job.job_id)}
              className="text-gray-400 hover:text-red-500 transition-colors"
              title="Delete"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </td>
    </tr>
  );
}

export default function TrainingList() {
  const [page, setPage] = useState(1);
  const { data, isLoading, error } = useTrainingJobs({ page, per_page: 20 });
  const deleteMutation = useDeleteTrainingJob();
  const cancelMutation = useCancelTrainingJob();

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this training job?')) {
      await deleteMutation.mutateAsync(id);
    }
  };

  const handleCancel = async (id: string) => {
    if (confirm('Are you sure you want to cancel this training job?')) {
      await cancelMutation.mutateAsync(id);
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
        Failed to load training jobs
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Training Jobs</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Monitor and manage your fine-tuning jobs
          </p>
        </div>
        <Link
          to="/training/new"
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Training
        </Link>
      </div>

      {/* Table */}
      {data?.jobs.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
          <Play className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No training jobs yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Start fine-tuning your first model
          </p>
          <Link
            to="/training/new"
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Training
          </Link>
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Job
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Method
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Loss
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {data?.jobs.map((job) => (
                <JobRow
                  key={job.job_id}
                  job={job}
                  onDelete={handleDelete}
                  onCancel={handleCancel}
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
