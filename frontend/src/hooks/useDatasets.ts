/**
 * Dataset hooks
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { datasetsApi } from '../lib/api';

export interface Dataset {
  dataset_id: string;
  user_id: string;
  name: string;
  description: string | null;
  format: 'jsonl' | 'csv' | 'parquet' | 'json';
  dataset_type: 'sft' | 'dpo' | 'orpo' | 'chat';
  file_path: string;
  file_size_bytes: number | null;
  total_examples: number | null;
  train_examples: number | null;
  validation_examples: number | null;
  avg_input_length: number | null;
  avg_output_length: number | null;
  is_validated: boolean;
  validation_errors: Array<{ row: number; column: string; message: string }> | null;
  validation_warnings: Array<{ row: number; column: string; message: string }> | null;
  status: 'pending' | 'validating' | 'ready' | 'error' | 'archived';
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface DatasetList {
  datasets: Dataset[];
  total: number;
  page: number;
  per_page: number;
}

export interface DatasetPreview {
  dataset_id: string;
  format: string;
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
  preview_rows: number;
}

export function useDatasets(params?: { page?: number; per_page?: number; status?: string }) {
  return useQuery<DatasetList>({
    queryKey: ['datasets', params],
    queryFn: async () => {
      const response = await datasetsApi.list(params);
      return response.data;
    },
  });
}

export function useDataset(id: string) {
  return useQuery<Dataset>({
    queryKey: ['dataset', id],
    queryFn: async () => {
      const response = await datasetsApi.get(id);
      return response.data;
    },
    enabled: !!id,
  });
}

export function useDatasetPreview(id: string, limit = 10) {
  return useQuery<DatasetPreview>({
    queryKey: ['dataset-preview', id, limit],
    queryFn: async () => {
      const response = await datasetsApi.preview(id, { limit });
      return response.data;
    },
    enabled: !!id,
  });
}

export function useUploadDataset() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await datasetsApi.create(formData);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}

export function useValidateDataset() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const response = await datasetsApi.validate(id);
      return response.data;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['dataset', id] });
    },
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      await datasetsApi.delete(id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}
