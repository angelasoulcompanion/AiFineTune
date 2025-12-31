import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../lib/api';

// Types
export interface Model {
  model_id: string;
  user_id: string;
  name: string;
  model_type: 'base' | 'lora' | 'merged';
  base_model_id: string;
  description?: string;
  file_path?: string;
  file_size_mb?: number;
  hf_repo_id?: string;
  is_pushed_to_hf: boolean;
  ollama_model_name?: string;
  is_imported_to_ollama: boolean;
  status: 'available' | 'downloading' | 'ready' | 'error';
  created_at: string;
  updated_at: string;
}

export interface HFModel {
  model_id: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  pipeline_tag?: string;
  last_modified?: string;
}

export interface HFModelInfo {
  model_id: string;
  author: string;
  sha: string;
  downloads: number;
  likes: number;
  tags: string[];
  pipeline_tag?: string;
  library_name?: string;
  size_bytes?: number;
  size_mb?: number;
  files: Array<{ filename: string; size?: number }>;
  gated: boolean;
  private: boolean;
}

export interface PopularModel {
  model_id: string;
  name: string;
  params: string;
  family: string;
  quantization?: string;
}

// Queries

export function useModels(params?: {
  page?: number;
  per_page?: number;
  model_type?: string;
  status?: string;
}) {
  return useQuery({
    queryKey: ['models', params],
    queryFn: async () => {
      const { data } = await api.get('/models', { params });
      return data as {
        models: Model[];
        total: number;
        page: number;
        per_page: number;
        pages: number;
      };
    },
  });
}

export function useModel(modelId: string) {
  return useQuery({
    queryKey: ['model', modelId],
    queryFn: async () => {
      const { data } = await api.get(`/models/${modelId}`);
      return data as Model;
    },
    enabled: !!modelId,
  });
}

export function useSearchHuggingFace(query: string, limit = 20) {
  return useQuery({
    queryKey: ['hf-search', query, limit],
    queryFn: async () => {
      const { data } = await api.get('/models/huggingface/search', {
        params: { query, limit },
      });
      return data as { models: HFModel[]; total: number };
    },
    enabled: query.length >= 2,
  });
}

export function useHFModelInfo(modelId: string) {
  return useQuery({
    queryKey: ['hf-info', modelId],
    queryFn: async () => {
      const { data } = await api.get(`/models/huggingface/info/${encodeURIComponent(modelId)}`);
      return data as HFModelInfo;
    },
    enabled: !!modelId,
  });
}

export function usePopularModels() {
  return useQuery({
    queryKey: ['popular-models'],
    queryFn: async () => {
      const { data } = await api.get('/models/huggingface/popular');
      return data as { models: PopularModel[] };
    },
  });
}

export function useBaseModelsForTraining() {
  return useQuery({
    queryKey: ['base-models-for-training'],
    queryFn: async () => {
      const { data } = await api.get('/models/for-training/base');
      return data as { models: Model[] };
    },
  });
}

export function useLoRAModels() {
  return useQuery({
    queryKey: ['lora-models'],
    queryFn: async () => {
      const { data } = await api.get('/models/for-training/lora');
      return data as { models: Model[] };
    },
  });
}

// Mutations

export function useDownloadModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: {
      hf_model_id: string;
      name: string;
      description?: string;
    }) => {
      const { data } = await api.post('/models/huggingface/download', params);
      return data as Model;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}

export function useCreateModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: {
      name: string;
      model_type: string;
      base_model_id: string;
      description?: string;
    }) => {
      const { data } = await api.post('/models', params);
      return data as Model;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (modelId: string) => {
      await api.delete(`/models/${modelId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}

export function usePushToHuggingFace() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: {
      modelId: string;
      repo_name: string;
      private?: boolean;
    }) => {
      const { modelId, ...body } = params;
      const { data } = await api.post(`/models/${modelId}/push-hf`, body);
      return data as Model & { hf_url: string };
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['model', variables.modelId] });
    },
  });
}

export function useImportToOllama() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { modelId: string; ollama_name: string }) => {
      const { modelId, ...body } = params;
      const { data } = await api.post(`/models/${modelId}/import-ollama`, body);
      return data as Model;
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['model', variables.modelId] });
    },
  });
}

export function useTestModel() {
  return useMutation({
    mutationFn: async (params: {
      modelId: string;
      prompt: string;
      max_tokens?: number;
    }) => {
      const { modelId, ...body } = params;
      const { data } = await api.post(`/models/${modelId}/test`, body);
      return data as { model: string; prompt: string; response: string };
    },
  });
}
