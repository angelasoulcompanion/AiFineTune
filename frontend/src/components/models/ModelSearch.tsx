import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  Download,
  Star,
  ArrowDownToLine,
  Sparkles,
  AlertCircle,
  CheckCircle,
  Loader2,
} from 'lucide-react';
import {
  useSearchHuggingFace,
  usePopularModels,
  useDownloadModel,
} from '../../hooks/useModels';
import type { HFModel, PopularModel } from '../../hooks/useModels';
import debounce from 'lodash.debounce';

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

interface PopularModelCardProps {
  model: PopularModel;
  onSelect: (modelId: string, name: string) => void;
  isDownloading: boolean;
}

function PopularModelCard({ model, onSelect, isDownloading }: PopularModelCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:border-primary-500 transition-colors">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-medium text-gray-900 dark:text-gray-100">{model.name}</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 font-mono">{model.model_id}</p>
          <div className="flex items-center gap-2 mt-2">
            <span className="text-xs px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded">
              {model.params}
            </span>
            <span className="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded">
              {model.family}
            </span>
          </div>
        </div>
        <button
          onClick={() => onSelect(model.model_id, model.name)}
          disabled={isDownloading}
          className="p-2 text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-lg transition-colors disabled:opacity-50"
        >
          {isDownloading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Download className="w-5 h-5" />
          )}
        </button>
      </div>
    </div>
  );
}

interface SearchResultCardProps {
  model: HFModel;
  onSelect: (modelId: string) => void;
  isDownloading: boolean;
}

function SearchResultCard({ model, onSelect, isDownloading }: SearchResultCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:border-primary-500 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-gray-900 dark:text-gray-100 truncate">{model.model_id}</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">by {model.author}</p>
          <div className="flex items-center gap-4 mt-2 text-sm text-gray-500 dark:text-gray-400">
            <span className="flex items-center gap-1">
              <ArrowDownToLine className="w-4 h-4" />
              {formatNumber(model.downloads)}
            </span>
            <span className="flex items-center gap-1">
              <Star className="w-4 h-4" />
              {formatNumber(model.likes)}
            </span>
          </div>
          {model.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {model.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="text-xs px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={() => onSelect(model.model_id)}
          disabled={isDownloading}
          className="p-2 text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-lg transition-colors disabled:opacity-50"
        >
          {isDownloading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Download className="w-5 h-5" />
          )}
        </button>
      </div>
    </div>
  );
}

export default function ModelSearch() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelName, setModelName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');

  const { data: popularModels } = usePopularModels();
  const { data: searchResults, isLoading: isSearching } = useSearchHuggingFace(debouncedQuery);
  const downloadMutation = useDownloadModel();

  // Debounce search
  const debouncedSetQuery = useCallback(
    debounce((query: string) => setDebouncedQuery(query), 500),
    []
  );

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchQuery(value);
    debouncedSetQuery(value);
  };

  const handleSelectModel = (modelId: string, name?: string) => {
    setSelectedModel(modelId);
    setModelName(name || modelId.split('/').pop() || modelId);
    setError('');
  };

  const handleDownload = async () => {
    if (!selectedModel || !modelName.trim()) {
      setError('Please enter a name for the model');
      return;
    }

    try {
      const model = await downloadMutation.mutateAsync({
        hf_model_id: selectedModel,
        name: modelName.trim(),
        description: description.trim() || undefined,
      });
      navigate(`/models/${model.model_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to download model');
    }
  };

  const handleCancel = () => {
    setSelectedModel(null);
    setModelName('');
    setDescription('');
    setError('');
  };

  // Show download dialog
  if (selectedModel) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Download Model
          </h2>

          {error && (
            <div className="mb-4 flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                HuggingFace Model
              </label>
              <p className="font-mono text-sm bg-gray-50 dark:bg-gray-700 px-3 py-2 rounded border border-gray-200 dark:border-gray-600">
                {selectedModel}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Display Name *
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="My Llama Model"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="Optional description..."
              />
            </div>

            <div className="flex gap-3 pt-4">
              <button
                onClick={handleCancel}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={handleDownload}
                disabled={downloadMutation.isPending}
                className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg transition-colors"
              >
                {downloadMutation.isPending ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" />
                    Download Model
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Add Model</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Search and download models from HuggingFace
        </p>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search HuggingFace models..."
          value={searchQuery}
          onChange={handleSearchChange}
          className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
        />
      </div>

      {/* Search Results */}
      {searchQuery.length >= 2 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <Search className="w-5 h-5" />
            Search Results
            {isSearching && <Loader2 className="w-4 h-4 animate-spin" />}
          </h2>

          {searchResults?.models.length === 0 && !isSearching ? (
            <p className="text-gray-500 dark:text-gray-400">No models found for "{searchQuery}"</p>
          ) : (
            <div className="grid gap-3">
              {searchResults?.models.map((model) => (
                <SearchResultCard
                  key={model.model_id}
                  model={model}
                  onSelect={handleSelectModel}
                  isDownloading={downloadMutation.isPending}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Popular Models */}
      {!searchQuery && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-yellow-500" />
            Recommended Models
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Pre-optimized models ready for fine-tuning (via Unsloth)
          </p>

          <div className="grid md:grid-cols-2 gap-3">
            {popularModels?.models.map((model) => (
              <PopularModelCard
                key={model.model_id}
                model={model}
                onSelect={handleSelectModel}
                isDownloading={downloadMutation.isPending}
              />
            ))}
          </div>
        </div>
      )}

      {/* Info */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div className="flex gap-3">
          <CheckCircle className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-700 dark:text-blue-300">
            <p className="font-medium">Need a HuggingFace Token?</p>
            <p>Some models require authentication. Add your token in Settings to access gated models.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
