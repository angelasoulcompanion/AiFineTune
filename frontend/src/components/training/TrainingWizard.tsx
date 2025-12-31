import { useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Database,
  Box,
  Zap,
  Settings,
  Play,
  ChevronRight,
  ChevronLeft,
  AlertCircle,
  Loader2,
  CheckCircle,
  Cloud,
} from 'lucide-react';
import { useDatasets } from '../../hooks/useDatasets';
import { useModels } from '../../hooks/useModels';
import { useCreateTrainingJob, useStartTrainingJob } from '../../hooks/useTraining';
import CloudConfigPanel from './CloudConfigPanel';
import clsx from 'clsx';

const steps = [
  { id: 1, title: 'Dataset', icon: Database },
  { id: 2, title: 'Model', icon: Box },
  { id: 3, title: 'Method', icon: Zap },
  { id: 4, title: 'Environment', icon: Cloud },
  { id: 5, title: 'Config', icon: Settings },
  { id: 6, title: 'Start', icon: Play },
];

const trainingMethods = [
  {
    id: 'lora',
    name: 'LoRA',
    description: 'Low-Rank Adaptation - Efficient fine-tuning',
    memory: 'Low',
    speed: 'Fast',
    recommended: true,
  },
  {
    id: 'qlora',
    name: 'QLoRA',
    description: '4-bit quantized LoRA - For larger models',
    memory: 'Very Low',
    speed: 'Medium',
  },
  {
    id: 'sft',
    name: 'Full Fine-Tuning',
    description: 'Full model training - Best quality',
    memory: 'High',
    speed: 'Slow',
  },
];

interface TrainingConfig {
  num_train_epochs: number;
  per_device_train_batch_size: number;
  gradient_accumulation_steps: number;
  learning_rate: number;
  warmup_ratio: number;
  max_seq_length: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
}

const defaultConfig: TrainingConfig = {
  num_train_epochs: 3,
  per_device_train_batch_size: 4,
  gradient_accumulation_steps: 4,
  learning_rate: 0.0002,
  warmup_ratio: 0.03,
  max_seq_length: 2048,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
};

export default function TrainingWizard() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [currentStep, setCurrentStep] = useState(1);
  const [selectedDataset, setSelectedDataset] = useState<string>(searchParams.get('dataset') || '');
  const [selectedModel, setSelectedModel] = useState<string>(searchParams.get('model') || '');
  const [selectedMethod, setSelectedMethod] = useState<string>('lora');
  const [executionEnv, setExecutionEnv] = useState<'local' | 'modal'>('local');
  const [selectedGPU, setSelectedGPU] = useState<string>('');
  const [jobName, setJobName] = useState('');
  const [config, setConfig] = useState<TrainingConfig>(defaultConfig);
  const [error, setError] = useState('');

  const { data: datasetsData, isLoading: datasetsLoading } = useDatasets({ per_page: 100 });
  const { data: modelsData, isLoading: modelsLoading } = useModels({ per_page: 100, status: 'ready' });

  const createMutation = useCreateTrainingJob();
  const startMutation = useStartTrainingJob();

  const readyDatasets = datasetsData?.datasets.filter((d) => d.status === 'ready') || [];
  const readyModels = modelsData?.models.filter((m) => m.status === 'ready') || [];

  const selectedDatasetObj = readyDatasets.find((d) => d.dataset_id === selectedDataset);
  const selectedModelObj = readyModels.find((m) => m.model_id === selectedModel);

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return !!selectedDataset;
      case 2:
        return !!selectedModel;
      case 3:
        return !!selectedMethod;
      case 4:
        return true; // Environment always valid (defaults to local)
      case 5:
        return !!jobName.trim();
      default:
        return true;
    }
  };

  const handleNext = () => {
    if (canProceed() && currentStep < 6) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleStart = async () => {
    setError('');

    try {
      // Create job
      const job = await createMutation.mutateAsync({
        dataset_id: selectedDataset,
        base_model_id: selectedModel,
        name: jobName.trim(),
        training_method: selectedMethod,
        execution_env: executionEnv,
        config,
      });

      // Start training
      await startMutation.mutateAsync(job.job_id);

      // Navigate to job detail
      navigate(`/training/${job.job_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start training');
    }
  };

  const isLoading = createMutation.isPending || startMutation.isPending;

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">New Training Job</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Configure and start fine-tuning
        </p>
      </div>

      {/* Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {steps.map((step, idx) => (
            <div key={step.id} className="flex items-center">
              <div
                className={clsx(
                  'flex items-center gap-2 px-3 py-2 rounded-lg transition-colors',
                  currentStep === step.id
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : currentStep > step.id
                      ? 'text-green-600 dark:text-green-400'
                      : 'text-gray-400'
                )}
              >
                {currentStep > step.id ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  <step.icon className="w-5 h-5" />
                )}
                <span className="text-sm font-medium hidden sm:block">{step.title}</span>
              </div>
              {idx < steps.length - 1 && (
                <ChevronRight className="w-4 h-4 text-gray-300 dark:text-gray-600 mx-2" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6 flex items-center gap-2 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Step Content */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 min-h-[400px]">
        {/* Step 1: Select Dataset */}
        {currentStep === 1 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Select Dataset
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Choose a validated dataset for training
            </p>

            {datasetsLoading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
              </div>
            ) : readyDatasets.length === 0 ? (
              <div className="text-center py-8">
                <Database className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600 dark:text-gray-400">No validated datasets available</p>
              </div>
            ) : (
              <div className="grid gap-3">
                {readyDatasets.map((dataset) => (
                  <label
                    key={dataset.dataset_id}
                    className={clsx(
                      'flex items-center p-4 border rounded-lg cursor-pointer transition-colors',
                      selectedDataset === dataset.dataset_id
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                    )}
                  >
                    <input
                      type="radio"
                      name="dataset"
                      value={dataset.dataset_id}
                      checked={selectedDataset === dataset.dataset_id}
                      onChange={(e) => setSelectedDataset(e.target.value)}
                      className="sr-only"
                    />
                    <div className="flex-1">
                      <p className="font-medium text-gray-900 dark:text-gray-100">{dataset.name}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {dataset.total_examples?.toLocaleString()} examples • {dataset.dataset_type.toUpperCase()}
                      </p>
                    </div>
                    {selectedDataset === dataset.dataset_id && (
                      <CheckCircle className="w-5 h-5 text-primary-500" />
                    )}
                  </label>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Step 2: Select Model */}
        {currentStep === 2 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Select Base Model
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Choose a model to fine-tune
            </p>

            {modelsLoading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
              </div>
            ) : readyModels.length === 0 ? (
              <div className="text-center py-8">
                <Box className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600 dark:text-gray-400">No models available</p>
              </div>
            ) : (
              <div className="grid gap-3">
                {readyModels.map((model) => (
                  <label
                    key={model.model_id}
                    className={clsx(
                      'flex items-center p-4 border rounded-lg cursor-pointer transition-colors',
                      selectedModel === model.model_id
                        ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                    )}
                  >
                    <input
                      type="radio"
                      name="model"
                      value={model.model_id}
                      checked={selectedModel === model.model_id}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="sr-only"
                    />
                    <div className="flex-1">
                      <p className="font-medium text-gray-900 dark:text-gray-100">{model.name}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                        {model.base_model_id}
                      </p>
                    </div>
                    {selectedModel === model.model_id && (
                      <CheckCircle className="w-5 h-5 text-primary-500" />
                    )}
                  </label>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Step 3: Select Method */}
        {currentStep === 3 && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Training Method
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Choose how to fine-tune the model
            </p>

            <div className="grid gap-3">
              {trainingMethods.map((method) => (
                <label
                  key={method.id}
                  className={clsx(
                    'flex items-start p-4 border rounded-lg cursor-pointer transition-colors',
                    selectedMethod === method.id
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  )}
                >
                  <input
                    type="radio"
                    name="method"
                    value={method.id}
                    checked={selectedMethod === method.id}
                    onChange={(e) => setSelectedMethod(e.target.value)}
                    className="sr-only"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-gray-900 dark:text-gray-100">{method.name}</p>
                      {method.recommended && (
                        <span className="text-xs px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded">
                          Recommended
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                      {method.description}
                    </p>
                    <div className="flex gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                      <span>Memory: {method.memory}</span>
                      <span>Speed: {method.speed}</span>
                    </div>
                  </div>
                  {selectedMethod === method.id && (
                    <CheckCircle className="w-5 h-5 text-primary-500 flex-shrink-0" />
                  )}
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Step 4: Environment */}
        {currentStep === 4 && (
          <CloudConfigPanel
            executionEnv={executionEnv}
            onExecutionEnvChange={setExecutionEnv}
            selectedGPU={selectedGPU}
            onGPUChange={setSelectedGPU}
            modelId={selectedModelObj?.base_model_id || ''}
            trainingMethod={selectedMethod}
            numSamples={selectedDatasetObj?.total_examples || 1000}
            numEpochs={config.num_train_epochs}
          />
        )}

        {/* Step 5: Configure */}
        {currentStep === 5 && (
          <div className="space-y-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Configuration
            </h2>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Job Name *
              </label>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="My Fine-tuning Job"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Epochs
                </label>
                <input
                  type="number"
                  value={config.num_train_epochs}
                  onChange={(e) => setConfig({ ...config, num_train_epochs: parseInt(e.target.value) || 1 })}
                  min="1"
                  max="100"
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={config.per_device_train_batch_size}
                  onChange={(e) => setConfig({ ...config, per_device_train_batch_size: parseInt(e.target.value) || 1 })}
                  min="1"
                  max="64"
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={config.learning_rate}
                  onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) || 0.0002 })}
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Max Sequence Length
                </label>
                <input
                  type="number"
                  value={config.max_seq_length}
                  onChange={(e) => setConfig({ ...config, max_seq_length: parseInt(e.target.value) || 2048 })}
                  min="128"
                  max="32768"
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
            </div>

            {(selectedMethod === 'lora' || selectedMethod === 'qlora') && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">LoRA Settings</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Rank (r)</label>
                    <input
                      type="number"
                      value={config.lora_r}
                      onChange={(e) => setConfig({ ...config, lora_r: parseInt(e.target.value) || 16 })}
                      min="1"
                      max="256"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Alpha</label>
                    <input
                      type="number"
                      value={config.lora_alpha}
                      onChange={(e) => setConfig({ ...config, lora_alpha: parseInt(e.target.value) || 32 })}
                      min="1"
                      max="512"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Dropout</label>
                    <input
                      type="number"
                      step="0.01"
                      value={config.lora_dropout}
                      onChange={(e) => setConfig({ ...config, lora_dropout: parseFloat(e.target.value) || 0.05 })}
                      min="0"
                      max="1"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Step 6: Review & Start */}
        {currentStep === 6 && (
          <div className="space-y-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Review & Start
            </h2>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Job Name</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">{jobName}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Dataset</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">{selectedDatasetObj?.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Model</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">{selectedModelObj?.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Method</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">{selectedMethod.toUpperCase()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Environment</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {executionEnv === 'modal' ? `Cloud (${selectedGPU})` : 'Local'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Epochs</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">{config.num_train_epochs}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500 dark:text-gray-400">Learning Rate</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">{config.learning_rate}</span>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <p className="text-sm text-blue-700 dark:text-blue-300">
                {executionEnv === 'modal'
                  ? `Training will run on Modal.com (${selectedGPU}). You'll be charged based on actual GPU usage time.`
                  : 'Training will start immediately on your local machine. You can monitor progress in real-time on the job detail page.'}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex justify-between mt-6">
        <button
          onClick={handleBack}
          disabled={currentStep === 1}
          className="inline-flex items-center gap-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50"
        >
          <ChevronLeft className="w-4 h-4" />
          Back
        </button>

        {currentStep < 6 ? (
          <button
            onClick={handleNext}
            disabled={!canProceed()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg transition-colors"
          >
            Next
            <ChevronRight className="w-4 h-4" />
          </button>
        ) : (
          <button
            onClick={handleStart}
            disabled={isLoading}
            className="inline-flex items-center gap-2 px-6 py-2 bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white rounded-lg transition-colors"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Starting...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Start Training
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
}
