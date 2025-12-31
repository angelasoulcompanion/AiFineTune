import { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeft,
  CheckCircle,
  AlertCircle,
  Download,
  Trash2,
  ExternalLink,
  Upload,
  Play,
  MessageSquare,
  Loader2,
  Box,
} from 'lucide-react';
import {
  useModel,
  useDeleteModel,
  usePushToHuggingFace,
  useImportToOllama,
  useTestModel,
} from '../hooks/useModels';
import clsx from 'clsx';

const statusConfig = {
  available: { icon: CheckCircle, color: 'text-gray-500', label: 'Available' },
  downloading: { icon: Download, color: 'text-blue-500', label: 'Downloading' },
  ready: { icon: CheckCircle, color: 'text-green-500', label: 'Ready' },
  error: { icon: AlertCircle, color: 'text-red-500', label: 'Error' },
};

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: model, isLoading, error } = useModel(id!);
  const deleteMutation = useDeleteModel();
  const pushMutation = usePushToHuggingFace();
  const ollamaMutation = useImportToOllama();
  const testMutation = useTestModel();

  const [showPushModal, setShowPushModal] = useState(false);
  const [showOllamaModal, setShowOllamaModal] = useState(false);
  const [showTestModal, setShowTestModal] = useState(false);

  const [repoName, setRepoName] = useState('');
  const [isPrivate, setIsPrivate] = useState(false);
  const [ollamaName, setOllamaName] = useState('');
  const [testPrompt, setTestPrompt] = useState('');
  const [testResponse, setTestResponse] = useState('');

  const handleDelete = async () => {
    if (id && confirm('Are you sure you want to delete this model?')) {
      await deleteMutation.mutateAsync(id);
      navigate('/models');
    }
  };

  const handlePush = async () => {
    if (!id || !repoName.trim()) return;

    try {
      await pushMutation.mutateAsync({
        modelId: id,
        repo_name: repoName.trim(),
        private: isPrivate,
      });
      setShowPushModal(false);
    } catch (err) {
      // Error handled by mutation
    }
  };

  const handleOllamaImport = async () => {
    if (!id || !ollamaName.trim()) return;

    try {
      await ollamaMutation.mutateAsync({
        modelId: id,
        ollama_name: ollamaName.trim(),
      });
      setShowOllamaModal(false);
    } catch (err) {
      // Error handled by mutation
    }
  };

  const handleTest = async () => {
    if (!id || !testPrompt.trim()) return;

    try {
      const result = await testMutation.mutateAsync({
        modelId: id,
        prompt: testPrompt.trim(),
      });
      setTestResponse(result.response);
    } catch (err: any) {
      setTestResponse(`Error: ${err.response?.data?.detail || 'Failed to test model'}`);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <p className="text-red-600 dark:text-red-400">Model not found</p>
        <Link to="/models" className="text-primary-600 hover:underline mt-2 inline-block">
          Back to models
        </Link>
      </div>
    );
  }

  const status = statusConfig[model.status];
  const StatusIcon = status.icon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            to="/models"
            className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-700 mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to models
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {model.name}
          </h1>
          {model.description && (
            <p className="text-gray-600 dark:text-gray-400 mt-1">{model.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {model.status === 'ready' && (
            <Link
              to={`/training/new?model=${model.model_id}`}
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
            {model.model_type.toUpperCase()}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Base Model</p>
          <p className="font-medium text-gray-900 dark:text-gray-100 mt-1 font-mono text-sm truncate">
            {model.base_model_id}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400">Size</p>
          <p className="font-medium text-gray-900 dark:text-gray-100 mt-1">
            {model.file_size_mb ? `${model.file_size_mb.toFixed(1)} MB` : '-'}
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Actions
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          {/* Push to HuggingFace */}
          <button
            onClick={() => setShowPushModal(true)}
            disabled={!model.file_path || model.is_pushed_to_hf}
            className="flex flex-col items-center gap-2 p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Upload className="w-6 h-6 text-orange-500" />
            <span className="font-medium text-gray-900 dark:text-gray-100">
              Push to HuggingFace
            </span>
            {model.is_pushed_to_hf && (
              <span className="text-xs text-green-500 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Pushed
              </span>
            )}
          </button>

          {/* Import to Ollama */}
          <button
            onClick={() => {
              setOllamaName(model.name.toLowerCase().replace(/\s+/g, '-'));
              setShowOllamaModal(true);
            }}
            disabled={!model.file_path || model.is_imported_to_ollama}
            className="flex flex-col items-center gap-2 p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Box className="w-6 h-6 text-blue-500" />
            <span className="font-medium text-gray-900 dark:text-gray-100">
              Import to Ollama
            </span>
            {model.is_imported_to_ollama && (
              <span className="text-xs text-green-500 flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Imported
              </span>
            )}
          </button>

          {/* Test Model */}
          <button
            onClick={() => setShowTestModal(true)}
            disabled={!model.is_imported_to_ollama}
            className="flex flex-col items-center gap-2 p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <MessageSquare className="w-6 h-6 text-green-500" />
            <span className="font-medium text-gray-900 dark:text-gray-100">
              Test with Prompt
            </span>
            {!model.is_imported_to_ollama && (
              <span className="text-xs text-gray-500">Import to Ollama first</span>
            )}
          </button>
        </div>
      </div>

      {/* External Links */}
      {model.hf_repo_id && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Links
          </h3>
          <a
            href={`https://huggingface.co/${model.hf_repo_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-primary-600 hover:underline"
          >
            <ExternalLink className="w-4 h-4" />
            View on HuggingFace
          </a>
        </div>
      )}

      {/* Push to HuggingFace Modal */}
      {showPushModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Push to HuggingFace
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Repository Name *
                </label>
                <input
                  type="text"
                  value={repoName}
                  onChange={(e) => setRepoName(e.target.value)}
                  placeholder="username/model-name"
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={isPrivate}
                  onChange={(e) => setIsPrivate(e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">Private repository</span>
              </label>
              <div className="flex gap-3 pt-2">
                <button
                  onClick={() => setShowPushModal(false)}
                  className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300"
                >
                  Cancel
                </button>
                <button
                  onClick={handlePush}
                  disabled={pushMutation.isPending || !repoName.trim()}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg"
                >
                  {pushMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Pushing...
                    </>
                  ) : (
                    'Push'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Import to Ollama Modal */}
      {showOllamaModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Import to Ollama
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Ollama Model Name *
                </label>
                <input
                  type="text"
                  value={ollamaName}
                  onChange={(e) => setOllamaName(e.target.value)}
                  placeholder="my-model"
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Use lowercase letters, numbers, and hyphens
                </p>
              </div>
              <div className="flex gap-3 pt-2">
                <button
                  onClick={() => setShowOllamaModal(false)}
                  className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300"
                >
                  Cancel
                </button>
                <button
                  onClick={handleOllamaImport}
                  disabled={ollamaMutation.isPending || !ollamaName.trim()}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg"
                >
                  {ollamaMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Importing...
                    </>
                  ) : (
                    'Import'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Model Modal */}
      {showTestModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-2xl w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Test Model: {model.ollama_model_name}
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Prompt
                </label>
                <textarea
                  value={testPrompt}
                  onChange={(e) => setTestPrompt(e.target.value)}
                  rows={3}
                  placeholder="Enter your prompt..."
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
              {testResponse && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Response
                  </label>
                  <pre className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-sm overflow-auto max-h-64 whitespace-pre-wrap">
                    {testResponse}
                  </pre>
                </div>
              )}
              <div className="flex gap-3 pt-2">
                <button
                  onClick={() => {
                    setShowTestModal(false);
                    setTestPrompt('');
                    setTestResponse('');
                  }}
                  className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300"
                >
                  Close
                </button>
                <button
                  onClick={handleTest}
                  disabled={testMutation.isPending || !testPrompt.trim()}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg"
                >
                  {testMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Generate
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
