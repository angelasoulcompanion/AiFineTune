import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../lib/api';

// Types
export interface HFTokenStatus {
  has_token: boolean;
  token_prefix: string | null;
  is_valid: boolean | null;
  hf_username: string | null;
  hf_name: string | null;
  can_write: boolean | null;
  error: string | null;
}

export interface HFTokenValidation {
  valid: boolean;
  username: string | null;
  name: string | null;
  email: string | null;
  orgs: string[];
  can_write: boolean;
  error: string | null;
}

export interface UserPreferences {
  theme?: 'light' | 'dark' | 'system';
  defaultTrainingMethod?: 'sft' | 'lora' | 'qlora' | 'dpo' | 'orpo';
  notifications?: boolean;
}

// Queries
export function useHFTokenStatus() {
  return useQuery({
    queryKey: ['hf-token-status'],
    queryFn: async () => {
      const { data } = await api.get('/auth/hf-token');
      return data as HFTokenStatus;
    },
  });
}

export function useHFTokenValidation() {
  return useQuery({
    queryKey: ['hf-token-validation'],
    queryFn: async () => {
      const { data } = await api.get('/auth/hf-token/validate');
      return data as HFTokenValidation;
    },
    enabled: false, // Only fetch on demand
  });
}

// Mutations
export function useSaveHFToken() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { hf_token: string; validate?: boolean }) => {
      const { data } = await api.post('/auth/hf-token', params);
      return data as HFTokenStatus;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['hf-token-status'] });
      queryClient.invalidateQueries({ queryKey: ['hf-token-validation'] });
    },
  });
}

export function useDeleteHFToken() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      await api.delete('/auth/hf-token');
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['hf-token-status'] });
      queryClient.invalidateQueries({ queryKey: ['hf-token-validation'] });
    },
  });
}

export function useValidateHFToken() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      const { data } = await api.get('/auth/hf-token/validate');
      return data as HFTokenValidation;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['hf-token-validation'] });
    },
  });
}

export function useChangePassword() {
  return useMutation({
    mutationFn: async (params: { current_password: string; new_password: string }) => {
      await api.post('/auth/change-password', params);
    },
  });
}

export function useUpdateProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { username?: string; preferences?: UserPreferences }) => {
      const { data } = await api.put('/auth/me', params);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user'] });
    },
  });
}
