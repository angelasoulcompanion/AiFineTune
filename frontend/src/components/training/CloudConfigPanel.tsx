import { useEffect } from 'react';
import {
  Cloud,
  Server,
  DollarSign,
  Clock,
  Cpu,
  AlertTriangle,
  CheckCircle,
  Loader2,
  Zap,
} from 'lucide-react';
import clsx from 'clsx';
import {
  useCloudGPUTypes,
  useCloudStatus,
  useCloudCostEstimate,
} from '../../hooks/useTraining';
import type { CloudGPUType } from '../../hooks/useTraining';

interface CloudConfigPanelProps {
  executionEnv: 'local' | 'modal';
  onExecutionEnvChange: (env: 'local' | 'modal') => void;
  selectedGPU: string;
  onGPUChange: (gpu: string) => void;
  modelId: string;
  trainingMethod: string;
  numSamples: number;
  numEpochs: number;
}

export default function CloudConfigPanel({
  executionEnv,
  onExecutionEnvChange,
  selectedGPU,
  onGPUChange,
  modelId,
  trainingMethod,
  numSamples,
  numEpochs,
}: CloudConfigPanelProps) {
  const { data: gpuData, isLoading: gpuLoading } = useCloudGPUTypes();
  const { data: cloudStatus, isLoading: statusLoading } = useCloudStatus();
  const costEstimateMutation = useCloudCostEstimate();

  // Fetch cost estimate when cloud is selected
  useEffect(() => {
    if (executionEnv === 'modal' && modelId && numSamples > 0) {
      costEstimateMutation.mutate({
        model_id: modelId,
        training_method: trainingMethod,
        num_samples: numSamples,
        num_epochs: numEpochs,
        gpu_type: selectedGPU || undefined,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [executionEnv, modelId, trainingMethod, numSamples, numEpochs, selectedGPU]);

  // Set default GPU when data loads
  useEffect(() => {
    if (gpuData && !selectedGPU) {
      onGPUChange(gpuData.recommended);
    }
  }, [gpuData, selectedGPU, onGPUChange]);

  const isCloudAvailable = cloudStatus?.modal_configured && cloudStatus?.modal_installed;

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
        Execution Environment
      </h3>

      {/* Environment Selection */}
      <div className="grid grid-cols-2 gap-4">
        {/* Local Option */}
        <button
          onClick={() => onExecutionEnvChange('local')}
          className={clsx(
            'flex flex-col items-center p-4 border rounded-lg transition-all',
            executionEnv === 'local'
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 ring-2 ring-primary-500'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
          )}
        >
          <Server className="w-8 h-8 mb-2 text-gray-600 dark:text-gray-400" />
          <span className="font-medium text-gray-900 dark:text-gray-100">Local</span>
          <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Use your machine
          </span>
          <span className="text-xs text-green-600 dark:text-green-400 mt-1">Free</span>
        </button>

        {/* Cloud Option */}
        <button
          onClick={() => isCloudAvailable && onExecutionEnvChange('modal')}
          disabled={!isCloudAvailable}
          className={clsx(
            'flex flex-col items-center p-4 border rounded-lg transition-all relative',
            executionEnv === 'modal'
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 ring-2 ring-primary-500'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300',
            !isCloudAvailable && 'opacity-50 cursor-not-allowed'
          )}
        >
          <Cloud className="w-8 h-8 mb-2 text-blue-500" />
          <span className="font-medium text-gray-900 dark:text-gray-100">Cloud (Modal)</span>
          <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            GPU on demand
          </span>
          <span className="text-xs text-blue-600 dark:text-blue-400 mt-1">Pay per use</span>
          {!isCloudAvailable && (
            <span className="absolute -top-2 -right-2 bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300 text-xs px-2 py-0.5 rounded-full">
              Not configured
            </span>
          )}
        </button>
      </div>

      {/* Cloud Status Warning */}
      {statusLoading ? (
        <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
          <Loader2 className="w-4 h-4 animate-spin" />
          Checking cloud status...
        </div>
      ) : !isCloudAvailable && (
        <div className="flex items-start gap-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg text-yellow-700 dark:text-yellow-300">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="font-medium">Modal.com not configured</p>
            <p className="text-yellow-600 dark:text-yellow-400">
              Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables to enable cloud training.
            </p>
          </div>
        </div>
      )}

      {/* GPU Selection (only for cloud) */}
      {executionEnv === 'modal' && isCloudAvailable && (
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Select GPU Type
          </h4>

          {gpuLoading ? (
            <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading GPU options...
            </div>
          ) : (
            <div className="space-y-2">
              {gpuData?.gpus.map((gpu: CloudGPUType) => (
                <label
                  key={gpu.id}
                  className={clsx(
                    'flex items-center p-3 border rounded-lg cursor-pointer transition-colors',
                    selectedGPU === gpu.id
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  )}
                >
                  <input
                    type="radio"
                    name="gpu"
                    value={gpu.id}
                    checked={selectedGPU === gpu.id}
                    onChange={(e) => onGPUChange(e.target.value)}
                    className="sr-only"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-gray-500" />
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {gpu.name}
                      </span>
                      {gpuData.recommended === gpu.id && (
                        <span className="text-xs px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded">
                          Recommended
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-4 mt-1 text-xs text-gray-500 dark:text-gray-400">
                      <span>{gpu.memory_gb} GB VRAM</span>
                      <span>${gpu.cost_per_hour.toFixed(2)}/hr</span>
                    </div>
                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                      Best for: {gpu.best_for.join(', ')}
                    </p>
                  </div>
                  {selectedGPU === gpu.id && (
                    <CheckCircle className="w-5 h-5 text-primary-500 flex-shrink-0" />
                  )}
                </label>
              ))}
            </div>
          )}

          {/* Cost Estimate */}
          {costEstimateMutation.isPending ? (
            <div className="flex items-center gap-2 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg text-gray-500 dark:text-gray-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              Calculating cost estimate...
            </div>
          ) : costEstimateMutation.data && (
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <h5 className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-3 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Cost Estimate
              </h5>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-blue-500" />
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Duration</p>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      ~{costEstimateMutation.data.estimate.estimated_minutes.toFixed(0)} min
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <DollarSign className="w-4 h-4 text-green-500" />
                  <div>
                    <p className="text-gray-500 dark:text-gray-400">Estimated Cost</p>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      ${costEstimateMutation.data.estimate.estimated_cost_usd.toFixed(2)}
                    </p>
                  </div>
                </div>
              </div>
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-3">
                Confidence: {costEstimateMutation.data.estimate.confidence} •
                Model size: {costEstimateMutation.data.model_size_b.toFixed(1)}B params
              </p>
            </div>
          )}
        </div>
      )}

      {/* Local Training Info */}
      {executionEnv === 'local' && (
        <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Local Training
          </h5>
          <ul className="text-sm text-gray-500 dark:text-gray-400 space-y-1">
            <li>• Uses your machine's GPU (CUDA/MPS) or CPU</li>
            <li>• No additional cost</li>
            <li>• Speed depends on your hardware</li>
            <li>• Model stays on your machine</li>
          </ul>
        </div>
      )}
    </div>
  );
}
