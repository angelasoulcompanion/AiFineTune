import { useParams, useNavigate, Link } from 'react-router-dom';
import { useState } from 'react';
import {
  ArrowLeft,
  CheckCircle,
  AlertCircle,
  Clock,
  Loader2,
  StopCircle,
  Trash2,
  Play,
  ExternalLink,
  Activity,
  BarChart3,
  Timer,
} from 'lucide-react';
import {
  useTrainingJob,
  useTrainingProgress,
  useTrainingMetrics,
  useCancelTrainingJob,
  useDeleteTrainingJob,
  useStartTrainingJob,
} from '../hooks/useTraining';
import clsx from 'clsx';
import LossChart from '../components/training/LossChart';

const statusConfig = {
  queued: { icon: Clock, color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-700', label: 'Queued', animate: false },
  preparing: { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-100 dark:bg-blue-900/30', label: 'Preparing', animate: true },
  training: { icon: Activity, color: 'text-blue-500', bg: 'bg-blue-100 dark:bg-blue-900/30', label: 'Training', animate: true },
  evaluating: { icon: Loader2, color: 'text-purple-500', bg: 'bg-purple-100 dark:bg-purple-900/30', label: 'Evaluating', animate: true },
  saving: { icon: Loader2, color: 'text-yellow-500', bg: 'bg-yellow-100 dark:bg-yellow-900/30', label: 'Saving', animate: true },
  completed: { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-100 dark:bg-green-900/30', label: 'Completed', animate: false },
  failed: { icon: AlertCircle, color: 'text-red-500', bg: 'bg-red-100 dark:bg-red-900/30', label: 'Failed', animate: false },
  cancelled: { icon: StopCircle, color: 'text-gray-500', bg: 'bg-gray-100 dark:bg-gray-700', label: 'Cancelled', animate: false },
};

export default function TrainingDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'overview' | 'charts' | 'logs'>('overview');

  const { data: job, isLoading, error } = useTrainingJob(id!);
  const { progress, logs, isConnected } = useTrainingProgress(id!);
  const { data: metricsData } = useTrainingMetrics(id!);
  const cancelMutation = useCancelTrainingJob();
  const deleteMutation = useDeleteTrainingJob();
  const startMutation = useStartTrainingJob();

  // Parse metrics for chart
  const lossHistory = metricsData?.history?.filter((h: Record<string, number>) => h.loss !== undefined)
    .map((h: Record<string, number>, i: number) => ({ step: h.step || i, loss: h.loss })) || [];
  const lrHistory = metricsData?.history?.filter((h: Record<string, number>) => h.lr !== undefined)
    .map((h: Record<string, number>, i: number) => ({ step: h.step || i, lr: h.lr })) || [];
  const memoryHistory = metricsData?.history?.filter((h: Record<string, number>) => h.memory_gb !== undefined)
    .map((h: Record<string, number>, i: number) => ({ step: h.step || i, memory_gb: h.memory_gb })) || [];

  const handleCancel = async () => {
    if (id && confirm('Are you sure you want to cancel this training?')) {
      await cancelMutation.mutateAsync(id);
    }
  };

  const handleDelete = async () => {
    if (id && confirm('Are you sure you want to delete this training job?')) {
      await deleteMutation.mutateAsync(id);
      navigate('/training');
    }
  };

  const handleStart = async () => {
    if (id) {
      await startMutation.mutateAsync(id);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-600 dark:text-red-400">Training job not found</p>
        <Link to="/training" className="text-primary-600 hover:underline mt-2 inline-block">
          Back to training jobs
        </Link>
      </div>
    );
  }

  const status = statusConfig[job.status];
  const StatusIcon = status.icon;
  const isRunning = ['preparing', 'training', 'evaluating', 'saving'].includes(job.status);

  // Use WebSocket progress if available, fallback to API data
  const currentProgress = progress?.progress ?? job.progress_percentage ?? 0;
  const currentStep = progress?.step ?? job.current_step ?? 0;
  const totalSteps = progress?.total_steps ?? job.total_steps ?? 0;
  const currentEpoch = progress?.epoch ?? job.current_epoch ?? 0;
  const totalEpochs = progress?.total_epochs ?? job.total_epochs ?? 0;
  const currentLoss = progress?.loss ?? job.current_loss;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            to="/training"
            className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-700 mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to training jobs
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {job.name}
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            {job.training_method.toUpperCase()} • {job.dataset_name} → {job.model_name}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {job.status === 'queued' && (
            <button
              onClick={handleStart}
              disabled={startMutation.isPending}
              className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg"
            >
              {startMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Start
            </button>
          )}
          {isRunning && (
            <button
              onClick={handleCancel}
              disabled={cancelMutation.isPending}
              className="inline-flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg"
            >
              <StopCircle className="w-4 h-4" />
              Cancel
            </button>
          )}
          {!isRunning && (
            <button
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
              className="p-2 text-gray-400 hover:text-red-500 transition-colors"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Status Card */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className={clsx('inline-flex items-center gap-2 px-3 py-1.5 rounded-full', status.bg)}>
              <StatusIcon className={clsx('w-4 h-4', status.color, status.animate && 'animate-spin')} />
              <span className={clsx('font-medium', status.color)}>{status.label}</span>
            </span>
            {isConnected && isRunning && (
              <span className="flex items-center gap-1 text-xs text-green-500">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                Live
              </span>
            )}
          </div>
          {job.completed_at && (
            <span className="text-sm text-gray-500 dark:text-gray-400">
              Completed {new Date(job.completed_at).toLocaleString()}
            </span>
          )}
        </div>

        {/* Progress Bar */}
        {(isRunning || job.status === 'completed') && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">
                Epoch {currentEpoch}/{totalEpochs} • Step {currentStep}/{totalSteps}
              </span>
              <span className="font-medium text-gray-900 dark:text-gray-100">
                {currentProgress.toFixed(1)}%
              </span>
            </div>
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 transition-all duration-300 ease-out"
                style={{ width: `${currentProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error Message */}
        {job.error_message && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-red-600 dark:text-red-400">{job.error_message}</p>
          </div>
        )}
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Current Loss</p>
          <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mt-1">
            {currentLoss !== undefined ? currentLoss.toFixed(4) : '-'}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Best Loss</p>
          <p className="text-2xl font-semibold text-green-600 dark:text-green-400 mt-1">
            {job.best_loss !== undefined ? job.best_loss.toFixed(4) : '-'}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Learning Rate</p>
          <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mt-1">
            {progress?.learning_rate?.toExponential(2) ?? job.learning_rate?.toExponential(2) ?? '-'}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Method</p>
          <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mt-1">
            {job.training_method.toUpperCase()}
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex space-x-8">
          <button
            onClick={() => setActiveTab('overview')}
            className={clsx(
              'py-3 px-1 border-b-2 font-medium text-sm transition-colors',
              activeTab === 'overview'
                ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            )}
          >
            <span className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Overview
            </span>
          </button>
          <button
            onClick={() => setActiveTab('charts')}
            className={clsx(
              'py-3 px-1 border-b-2 font-medium text-sm transition-colors',
              activeTab === 'charts'
                ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            )}
          >
            <span className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Charts
              {lossHistory.length > 0 && (
                <span className="bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 text-xs px-2 py-0.5 rounded-full">
                  {lossHistory.length}
                </span>
              )}
            </span>
          </button>
          <button
            onClick={() => setActiveTab('logs')}
            className={clsx(
              'py-3 px-1 border-b-2 font-medium text-sm transition-colors',
              activeTab === 'logs'
                ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            )}
          >
            <span className="flex items-center gap-2">
              <Timer className="w-4 h-4" />
              Logs
              {logs.length > 0 && (
                <span className="bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 text-xs px-2 py-0.5 rounded-full">
                  {logs.length}
                </span>
              )}
            </span>
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <>
          {/* Configuration */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Configuration
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400">Epochs</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {job.config?.training?.num_train_epochs ?? '-'}
                </p>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Batch Size</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {job.config?.training?.per_device_train_batch_size ?? '-'}
                </p>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Learning Rate</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {job.config?.training?.learning_rate ?? '-'}
                </p>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Max Seq Length</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {job.config?.training?.max_seq_length ?? '-'}
                </p>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">LR Scheduler</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {(job.config?.training as Record<string, unknown>)?.lr_scheduler_type as string ?? 'cosine'}
                </p>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Optimizer</span>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {(job.config?.training as Record<string, unknown>)?.optim as string ?? 'adamw_8bit'}
                </p>
              </div>
              {job.config?.lora && (
                <>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">LoRA Rank</span>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {job.config.lora.lora_r ?? '-'}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">LoRA Alpha</span>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {job.config.lora.lora_alpha ?? '-'}
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Output Model */}
          {job.output_model_id && (
            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="font-medium text-green-700 dark:text-green-300">
                    Training Complete!
                  </span>
                </div>
                <Link
                  to={`/models/${job.output_model_id}`}
                  className="inline-flex items-center gap-1 text-green-600 dark:text-green-400 hover:underline"
                >
                  View Output Model
                  <ExternalLink className="w-4 h-4" />
                </Link>
              </div>
            </div>
          )}
        </>
      )}

      {/* Charts Tab */}
      {activeTab === 'charts' && (
        <div className="space-y-6">
          {lossHistory.length > 0 ? (
            <LossChart
              lossHistory={lossHistory}
              lrHistory={lrHistory}
              memoryHistory={memoryHistory}
            />
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                No Training Data Yet
              </h3>
              <p className="text-gray-500 dark:text-gray-400">
                Charts will appear once training starts and metrics are collected.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Logs Tab */}
      {activeTab === 'logs' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Training Logs
          </h3>
          {logs.length > 0 ? (
            <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm max-h-96 overflow-y-auto">
              {logs.map((log, i) => (
                <div
                  key={i}
                  className={clsx(
                    'py-0.5',
                    log.level === 'error' && 'text-red-400',
                    log.level === 'warning' && 'text-yellow-400',
                    log.level === 'info' && 'text-green-400'
                  )}
                >
                  <span className="text-gray-500">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>{' '}
                  {log.message}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              No logs yet. Logs will appear when training starts.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
