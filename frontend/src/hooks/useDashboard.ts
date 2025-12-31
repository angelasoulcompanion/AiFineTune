import { useQuery } from '@tanstack/react-query';
import api from '../lib/api';

export interface DashboardStats {
  datasets: number;
  models: number;
  training_jobs: number;
  success_rate: number | null;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
}

export interface RecentJob {
  job_id: string;
  name: string;
  status: string;
  training_method: string;
  execution_env: string;
  progress: number | null;
  loss: number | null;
  dataset_name: string | null;
  model_name: string | null;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface MethodUsage {
  method: string;
  count: number;
}

export interface DashboardData {
  stats: DashboardStats;
  status_breakdown: Record<string, number>;
  method_usage: MethodUsage[];
  recent_jobs: RecentJob[];
}

export interface JobsPerDay {
  date: string;
  count: number;
}

export interface TrainingTimeByMethod {
  method: string;
  minutes: number;
}

export interface AvgLossByMethod {
  method: string;
  loss: number | null;
}

export interface EnvUsage {
  env: string;
  count: number;
}

export interface AnalyticsData {
  jobs_per_day: JobsPerDay[];
  training_time_by_method: TrainingTimeByMethod[];
  avg_loss_by_method: AvgLossByMethod[];
  env_usage: EnvUsage[];
}

export function useDashboardStats() {
  return useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const { data } = await api.get('/dashboard/stats');
      return data as DashboardData;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });
}

export function useDashboardAnalytics(days: number = 30) {
  return useQuery({
    queryKey: ['dashboard-analytics', days],
    queryFn: async () => {
      const { data } = await api.get(`/dashboard/analytics?days=${days}`);
      return data as AnalyticsData;
    },
  });
}
