import { Link } from 'react-router-dom';
import {
  Database,
  Box,
  Play,
  TrendingUp,
  CheckCircle,
  XCircle,
  Clock,
  Cloud,
  Server,
  Loader2,
  ArrowRight,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { useAuth } from '../contexts/AuthContext';
import { useDashboardStats, useDashboardAnalytics } from '../hooks/useDashboard';
import clsx from 'clsx';

const trainingMethods = [
  {
    id: 'sft',
    name: 'SFT',
    description: 'Supervised Fine-Tuning - Train on instruction-output pairs',
    complexity: 'Easy',
    memory: 'High',
  },
  {
    id: 'lora',
    name: 'LoRA',
    description: 'Low-Rank Adaptation - Efficient fine-tuning with low-rank matrices',
    complexity: 'Easy',
    memory: 'Low',
  },
  {
    id: 'qlora',
    name: 'QLoRA',
    description: '4-bit quantized LoRA for consumer GPUs',
    complexity: 'Easy',
    memory: 'Very Low',
  },
  {
    id: 'dpo',
    name: 'DPO',
    description: 'Direct Preference Optimization - Align model with preferences',
    complexity: 'Medium',
    memory: 'Medium',
  },
  {
    id: 'orpo',
    name: 'ORPO',
    description: 'Odds Ratio Preference Optimization - Single-stage alignment',
    complexity: 'Medium',
    memory: 'Medium',
  },
];

const COLORS = ['#8b5cf6', '#22c55e', '#3b82f6', '#f59e0b', '#ef4444'];

const statusColors: Record<string, string> = {
  completed: 'text-green-600 dark:text-green-400',
  failed: 'text-red-600 dark:text-red-400',
  training: 'text-purple-600 dark:text-purple-400',
  preparing: 'text-blue-600 dark:text-blue-400',
  queued: 'text-gray-600 dark:text-gray-400',
  cancelled: 'text-orange-600 dark:text-orange-400',
};

const statusIcons: Record<string, typeof CheckCircle> = {
  completed: CheckCircle,
  failed: XCircle,
  training: Loader2,
  preparing: Clock,
  queued: Clock,
  cancelled: XCircle,
};

export default function Dashboard() {
  const { user } = useAuth();
  const { data: statsData, isLoading: statsLoading } = useDashboardStats();
  const { data: analyticsData } = useDashboardAnalytics(30);

  const stats = statsData?.stats;
  const recentJobs = statsData?.recent_jobs || [];
  const methodUsage = statsData?.method_usage || [];

  return (
    <div className="space-y-6">
      {/* Welcome */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Welcome, {user?.username}!
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Fine-tune LLM models with ease
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-blue-500">
              <Database className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Datasets</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {statsLoading ? '-' : stats?.datasets || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-green-500">
              <Box className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Models</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {statsLoading ? '-' : stats?.models || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-purple-500">
              <Play className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Training Jobs</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {statsLoading ? '-' : stats?.training_jobs || 0}
              </p>
              {stats?.running_jobs ? (
                <p className="text-xs text-purple-600 dark:text-purple-400">
                  {stats.running_jobs} running
                </p>
              ) : null}
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-lg bg-orange-500">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {statsLoading ? '-' : stats?.success_rate != null ? `${stats.success_rate}%` : '-'}
              </p>
              {stats && (stats.completed_jobs > 0 || stats.failed_jobs > 0) && (
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {stats.completed_jobs} completed, {stats.failed_jobs} failed
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Quick actions */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Link
            to="/datasets/new"
            className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
          >
            <Database className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span className="font-medium text-blue-600 dark:text-blue-400">
              Upload Dataset
            </span>
          </Link>
          <Link
            to="/models/new"
            className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors"
          >
            <Box className="w-5 h-5 text-green-600 dark:text-green-400" />
            <span className="font-medium text-green-600 dark:text-green-400">
              Add Model
            </span>
          </Link>
          <Link
            to="/training/new"
            className="flex items-center gap-3 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
          >
            <Play className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <span className="font-medium text-purple-600 dark:text-purple-400">
              Start Training
            </span>
          </Link>
        </div>
      </div>

      {/* Analytics Charts */}
      {(methodUsage.length > 0 || analyticsData?.env_usage.length) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Method Usage Chart */}
          {methodUsage.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Training Methods Usage
              </h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={methodUsage}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis
                    dataKey="method"
                    tick={{ fill: '#9ca3af', fontSize: 12 }}
                    tickFormatter={(value) => value.toUpperCase()}
                  />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                    labelStyle={{ color: '#f3f4f6' }}
                  />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Environment Usage Pie Chart */}
          {analyticsData?.env_usage && analyticsData.env_usage.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Execution Environment
              </h2>
              <div className="flex items-center justify-center gap-8">
                <ResponsiveContainer width={150} height={150}>
                  <PieChart>
                    <Pie
                      data={analyticsData.env_usage as unknown as Array<{ env: string; count: number; [key: string]: string | number }>}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={60}
                      dataKey="count"
                      nameKey="env"
                    >
                      {analyticsData.env_usage.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div className="space-y-2">
                  {analyticsData.env_usage.map((item, index) => (
                    <div key={item.env} className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: COLORS[index % COLORS.length] }}
                      />
                      <span className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-1">
                        {item.env === 'modal' ? (
                          <Cloud className="w-4 h-4" />
                        ) : (
                          <Server className="w-4 h-4" />
                        )}
                        {item.env === 'modal' ? 'Cloud' : 'Local'}: {item.count}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recent jobs */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Recent Training Jobs
          </h2>
          {recentJobs.length > 0 && (
            <Link
              to="/training"
              className="text-sm text-primary-600 dark:text-primary-400 hover:underline flex items-center gap-1"
            >
              View all <ArrowRight className="w-4 h-4" />
            </Link>
          )}
        </div>

        {statsLoading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
          </div>
        ) : recentJobs.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <Play className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No training jobs yet</p>
            <p className="text-sm">Start by uploading a dataset and selecting a model</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-3 font-medium">Job</th>
                  <th className="pb-3 font-medium">Method</th>
                  <th className="pb-3 font-medium">Env</th>
                  <th className="pb-3 font-medium">Status</th>
                  <th className="pb-3 font-medium">Progress</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                {recentJobs.map((job) => {
                  const StatusIcon = statusIcons[job.status] || Clock;
                  return (
                    <tr key={job.job_id} className="text-sm">
                      <td className="py-3">
                        <Link
                          to={`/training/${job.job_id}`}
                          className="text-primary-600 dark:text-primary-400 hover:underline font-medium"
                        >
                          {job.name}
                        </Link>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {job.model_name}
                        </p>
                      </td>
                      <td className="py-3">
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs font-medium">
                          {job.training_method.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3">
                        {job.execution_env === 'modal' ? (
                          <Cloud className="w-4 h-4 text-blue-500" />
                        ) : (
                          <Server className="w-4 h-4 text-gray-500" />
                        )}
                      </td>
                      <td className="py-3">
                        <span className={clsx('flex items-center gap-1', statusColors[job.status])}>
                          <StatusIcon
                            className={clsx('w-4 h-4', job.status === 'training' && 'animate-spin')}
                          />
                          {job.status}
                        </span>
                      </td>
                      <td className="py-3">
                        {job.progress !== null ? (
                          <div className="flex items-center gap-2">
                            <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-primary-500 rounded-full transition-all"
                                style={{ width: `${job.progress}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {job.progress.toFixed(0)}%
                            </span>
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Training methods */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Supported Training Methods
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {trainingMethods.map((method) => (
            <div
              key={method.id}
              className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-primary-500 dark:hover:border-primary-500 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                  {method.name}
                </h3>
                <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-gray-600 dark:text-gray-400">
                  {method.complexity}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {method.description}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                Memory: {method.memory}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
