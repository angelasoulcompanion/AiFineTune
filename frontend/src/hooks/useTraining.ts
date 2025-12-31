import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState, useCallback } from 'react';
import api from '../lib/api';

// Types
export interface TrainingConfig {
  training?: {
    num_train_epochs?: number;
    per_device_train_batch_size?: number;
    gradient_accumulation_steps?: number;
    learning_rate?: number;
    warmup_ratio?: number;
    weight_decay?: number;
    max_seq_length?: number;
  };
  lora?: {
    lora_r?: number;
    lora_alpha?: number;
    lora_dropout?: number;
    target_modules?: string[];
  };
}

export interface TrainingJob {
  job_id: string;
  user_id: string;
  dataset_id: string;
  base_model_id: string;
  name: string;
  training_method: 'sft' | 'lora' | 'qlora' | 'dpo' | 'orpo';
  execution_env: 'local' | 'modal' | 'hf_spaces';
  status: 'queued' | 'preparing' | 'training' | 'evaluating' | 'saving' | 'completed' | 'failed' | 'cancelled';
  config?: TrainingConfig;
  current_epoch?: number;
  total_epochs?: number;
  current_step?: number;
  total_steps?: number;
  progress_percentage?: number;
  current_loss?: number;
  best_loss?: number;
  learning_rate?: number;
  training_metrics?: {
    latest?: Record<string, number>;
    history?: Array<Record<string, any>>;
  };
  output_model_id?: string;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;

  // Joined
  dataset_name?: string;
  model_name?: string;
}

export interface TrainingTemplate {
  template_id: string;
  name: string;
  description: string;
  training_method: string;
  config: TrainingConfig;
}

export interface TrainingProgress {
  type: 'progress' | 'status' | 'metrics' | 'log' | 'completed' | 'failed' | 'cancelled';
  job_id: string;
  epoch?: number;
  total_epochs?: number;
  step?: number;
  total_steps?: number;
  progress?: number;
  loss?: number;
  learning_rate?: number;
  status?: string;
  message?: string;
  metrics?: Record<string, number>;
  level?: string;
  error?: string;
  output_model_id?: string;
}

// Queries

export function useTrainingJobs(params?: {
  page?: number;
  per_page?: number;
  status?: string;
}) {
  return useQuery({
    queryKey: ['training-jobs', params],
    queryFn: async () => {
      const { data } = await api.get('/training', { params });
      return data as {
        jobs: TrainingJob[];
        total: number;
        page: number;
        per_page: number;
        pages: number;
      };
    },
  });
}

export function useTrainingJob(jobId: string) {
  return useQuery({
    queryKey: ['training-job', jobId],
    queryFn: async () => {
      const { data } = await api.get(`/training/${jobId}`);
      return data as TrainingJob;
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      // Refetch every 5s for running jobs
      const status = query.state.data?.status;
      if (status && ['queued', 'preparing', 'training', 'evaluating', 'saving'].includes(status)) {
        return 5000;
      }
      return false;
    },
  });
}

export function useTrainingTemplates() {
  return useQuery({
    queryKey: ['training-templates'],
    queryFn: async () => {
      const { data } = await api.get('/training/templates');
      return data as { templates: TrainingTemplate[] };
    },
  });
}

export function useTrainingStatus(jobId: string) {
  return useQuery({
    queryKey: ['training-status', jobId],
    queryFn: async () => {
      const { data } = await api.get(`/training/${jobId}/status`);
      return data as {
        job_id: string;
        status: string;
        progress_percentage: number;
        current_epoch?: number;
        total_epochs?: number;
        current_step?: number;
        total_steps?: number;
        current_loss?: number;
        best_loss?: number;
      };
    },
    enabled: !!jobId,
    refetchInterval: 3000, // Every 3 seconds
  });
}

export function useTrainingMetrics(jobId: string) {
  return useQuery({
    queryKey: ['training-metrics', jobId],
    queryFn: async () => {
      const { data } = await api.get(`/training/${jobId}/metrics`);
      return data as {
        job_id: string;
        latest: Record<string, number>;
        history: Array<Record<string, any>>;
      };
    },
    enabled: !!jobId,
  });
}

// Mutations

export function useCreateTrainingJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: {
      dataset_id: string;
      base_model_id: string;
      name: string;
      training_method: string;
      execution_env?: string;
      config?: {
        num_train_epochs?: number;
        per_device_train_batch_size?: number;
        gradient_accumulation_steps?: number;
        learning_rate?: number;
        warmup_ratio?: number;
        weight_decay?: number;
        max_seq_length?: number;
        lora_r?: number;
        lora_alpha?: number;
        lora_dropout?: number;
      };
    }) => {
      const { data } = await api.post('/training', params);
      return data as TrainingJob;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-jobs'] });
    },
  });
}

export function useStartTrainingJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (jobId: string) => {
      const { data } = await api.post(`/training/${jobId}/start`);
      return data as TrainingJob;
    },
    onSuccess: (_, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['training-job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['training-jobs'] });
    },
  });
}

export function useCancelTrainingJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (jobId: string) => {
      const { data } = await api.post(`/training/${jobId}/cancel`);
      return data as TrainingJob;
    },
    onSuccess: (_, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['training-job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['training-jobs'] });
    },
  });
}

export function useDeleteTrainingJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (jobId: string) => {
      await api.delete(`/training/${jobId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-jobs'] });
    },
  });
}

// ==================== Cloud Training Hooks ====================

export interface CloudGPUType {
  id: string;
  name: string;
  memory_gb: number;
  cost_per_hour: number;
  best_for: string[];
}

export interface CloudCostEstimate {
  gpu_type: string;
  gpu_memory_gb: number;
  estimated_minutes: number;
  estimated_cost_usd: number;
  cost_per_hour: number;
  best_for: string[];
  confidence: string;
}

export interface CloudStatus {
  modal_configured: boolean;
  modal_token_set: boolean;
  modal_secret_set: boolean;
  modal_installed?: boolean;
  modal_version?: string;
  error?: string;
}

export function useCloudGPUTypes() {
  return useQuery({
    queryKey: ['cloud-gpu-types'],
    queryFn: async () => {
      const { data } = await api.get('/training/cloud/gpu-types');
      return data as {
        gpus: CloudGPUType[];
        recommended: string;
        provider: string;
      };
    },
  });
}

export function useCloudStatus() {
  return useQuery({
    queryKey: ['cloud-status'],
    queryFn: async () => {
      const { data } = await api.get('/training/cloud/status');
      return data as CloudStatus;
    },
  });
}

export function useCloudCostEstimate() {
  return useMutation({
    mutationFn: async (params: {
      model_id: string;
      training_method: string;
      num_samples: number;
      num_epochs: number;
      gpu_type?: string;
    }) => {
      const { data } = await api.post('/training/cloud/estimate', params);
      return data as {
        estimate: CloudCostEstimate;
        model_size_b: number;
        recommended_gpu: string;
      };
    },
  });
}

export function useLRSchedulerTypes() {
  return useQuery({
    queryKey: ['lr-scheduler-types'],
    queryFn: async () => {
      const { data } = await api.get('/training/scheduler-types');
      return data as {
        schedulers: Array<{
          id: string;
          name: string;
          description: string;
          best_for: string;
        }>;
        recommended: string;
      };
    },
  });
}

export function useOptimizerTypes() {
  return useQuery({
    queryKey: ['optimizer-types'],
    queryFn: async () => {
      const { data } = await api.get('/training/optimizer-types');
      return data as {
        optimizers: Array<{
          id: string;
          name: string;
          description: string;
          best_for: string;
        }>;
        recommended: string;
      };
    },
  });
}

// WebSocket hook for real-time progress
export function useTrainingProgress(jobId: string | null) {
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [logs, setLogs] = useState<Array<{ level: string; message: string; timestamp: string }>>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const queryClient = useQueryClient();

  const connect = useCallback(() => {
    if (!jobId) return;

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/training/${jobId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as TrainingProgress;
        setProgress(data);

        // Add to logs if it's a log message
        if (data.type === 'log') {
          setLogs((prev) => [
            ...prev.slice(-99), // Keep last 100 logs
            {
              level: data.level || 'info',
              message: data.message || '',
              timestamp: new Date().toISOString(),
            },
          ]);
        }

        // Invalidate queries on completion/failure
        if (['completed', 'failed', 'cancelled'].includes(data.type)) {
          queryClient.invalidateQueries({ queryKey: ['training-job', jobId] });
          queryClient.invalidateQueries({ queryKey: ['training-jobs'] });
          queryClient.invalidateQueries({ queryKey: ['models'] });
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message', e);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = () => {
      setIsConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [jobId, queryClient]);

  useEffect(() => {
    const cleanup = connect();
    return () => {
      cleanup?.();
      wsRef.current?.close();
    };
  }, [connect]);

  return { progress, logs, isConnected };
}
