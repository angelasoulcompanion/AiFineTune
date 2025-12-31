/**
 * Advanced Training Configuration Panel
 */
import { useState } from 'react';
import { ChevronDown, ChevronUp, Info, Zap, Cpu, Timer } from 'lucide-react';
import clsx from 'clsx';

interface AdvancedConfigPanelProps {
  config: {
    // Training params
    num_train_epochs: number;
    per_device_train_batch_size: number;
    gradient_accumulation_steps: number;
    learning_rate: number;
    warmup_ratio: number;
    weight_decay: number;
    max_seq_length: number;
    // LR Scheduler
    lr_scheduler_type: string;
    warmup_steps?: number;
    num_cycles?: number;
    // LoRA
    lora_r: number;
    lora_alpha: number;
    lora_dropout: number;
    // Advanced
    gradient_checkpointing: boolean;
    optim: string;
    logging_steps: number;
    save_steps: number;
    eval_steps?: number;
  };
  onChange: (config: AdvancedConfigPanelProps['config']) => void;
  trainingMethod: string;
  datasetSize?: number;
}

const LR_SCHEDULERS = [
  { id: 'cosine', name: 'Cosine', description: 'Recommended for most tasks' },
  { id: 'linear', name: 'Linear', description: 'Linear decay to 0' },
  { id: 'cosine_with_restarts', name: 'Cosine with Restarts', description: 'Periodic warm restarts' },
  { id: 'polynomial', name: 'Polynomial', description: 'Polynomial decay' },
  { id: 'constant', name: 'Constant', description: 'Fixed learning rate' },
  { id: 'constant_with_warmup', name: 'Constant + Warmup', description: 'Warmup then constant' },
];

const OPTIMIZERS = [
  { id: 'adamw_8bit', name: 'AdamW 8-bit', description: 'Memory-efficient, recommended' },
  { id: 'adamw_torch', name: 'AdamW', description: 'Standard PyTorch AdamW' },
  { id: 'adamw_torch_fused', name: 'AdamW Fused', description: 'Faster on CUDA' },
  { id: 'adafactor', name: 'Adafactor', description: 'Very memory efficient' },
  { id: 'sgd', name: 'SGD', description: 'Simple gradient descent' },
];

export default function AdvancedConfigPanel({
  config,
  onChange,
  trainingMethod,
  datasetSize,
}: AdvancedConfigPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeSection, setActiveSection] = useState<'training' | 'lora' | 'scheduler' | 'advanced'>('training');

  const handleChange = (field: string, value: number | string | boolean) => {
    onChange({ ...config, [field]: value });
  };

  // Calculate estimated steps
  const effectiveBatchSize = config.per_device_train_batch_size * config.gradient_accumulation_steps;
  const stepsPerEpoch = datasetSize ? Math.ceil(datasetSize / effectiveBatchSize) : 0;
  const totalSteps = stepsPerEpoch * config.num_train_epochs;

  const showLoRA = ['lora', 'qlora', 'dpo', 'orpo'].includes(trainingMethod);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Zap className="w-5 h-5 text-primary-500" />
          <span className="font-medium text-gray-900 dark:text-gray-100">
            Advanced Configuration
          </span>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            LR scheduler, optimizer, LoRA settings
          </span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="border-t border-gray-200 dark:border-gray-700">
          {/* Section Tabs */}
          <div className="flex border-b border-gray-200 dark:border-gray-700 px-4">
            {['training', showLoRA && 'lora', 'scheduler', 'advanced'].filter(Boolean).map((section) => (
              <button
                key={section as string}
                onClick={() => setActiveSection(section as typeof activeSection)}
                className={clsx(
                  'px-4 py-2 text-sm font-medium border-b-2 transition-colors',
                  activeSection === section
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                )}
              >
                {section === 'training' && 'Training'}
                {section === 'lora' && 'LoRA'}
                {section === 'scheduler' && 'LR Scheduler'}
                {section === 'advanced' && 'Advanced'}
              </button>
            ))}
          </div>

          <div className="p-6">
            {/* Training Section */}
            {activeSection === 'training' && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Epochs
                  </label>
                  <input
                    type="number"
                    value={config.num_train_epochs}
                    onChange={(e) => handleChange('num_train_epochs', parseInt(e.target.value) || 1)}
                    min={1}
                    max={100}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={config.per_device_train_batch_size}
                    onChange={(e) => handleChange('per_device_train_batch_size', parseInt(e.target.value) || 1)}
                    min={1}
                    max={64}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Gradient Accum
                  </label>
                  <input
                    type="number"
                    value={config.gradient_accumulation_steps}
                    onChange={(e) => handleChange('gradient_accumulation_steps', parseInt(e.target.value) || 1)}
                    min={1}
                    max={64}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Learning Rate
                  </label>
                  <input
                    type="text"
                    value={config.learning_rate}
                    onChange={(e) => handleChange('learning_rate', parseFloat(e.target.value) || 2e-4)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Warmup Ratio
                  </label>
                  <input
                    type="number"
                    value={config.warmup_ratio}
                    onChange={(e) => handleChange('warmup_ratio', parseFloat(e.target.value) || 0)}
                    min={0}
                    max={1}
                    step={0.01}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Seq Length
                  </label>
                  <select
                    value={config.max_seq_length}
                    onChange={(e) => handleChange('max_seq_length', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    <option value={512}>512</option>
                    <option value={1024}>1024</option>
                    <option value={2048}>2048</option>
                    <option value={4096}>4096</option>
                    <option value={8192}>8192</option>
                  </select>
                </div>
              </div>
            )}

            {/* LoRA Section */}
            {activeSection === 'lora' && showLoRA && (
              <div className="space-y-4">
                <div className="flex items-start gap-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    LoRA (Low-Rank Adaptation) trains only a small subset of parameters,
                    making training faster and more memory efficient.
                  </p>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      LoRA Rank (r)
                      <span className="text-gray-400 ml-1">(8-64)</span>
                    </label>
                    <input
                      type="number"
                      value={config.lora_r}
                      onChange={(e) => handleChange('lora_r', parseInt(e.target.value) || 16)}
                      min={1}
                      max={256}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                    <p className="text-xs text-gray-500 mt-1">Higher = more capacity, more memory</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      LoRA Alpha
                      <span className="text-gray-400 ml-1">(usually 2x rank)</span>
                    </label>
                    <input
                      type="number"
                      value={config.lora_alpha}
                      onChange={(e) => handleChange('lora_alpha', parseInt(e.target.value) || 32)}
                      min={1}
                      max={512}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                    <p className="text-xs text-gray-500 mt-1">Scaling factor for LoRA</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      LoRA Dropout
                    </label>
                    <input
                      type="number"
                      value={config.lora_dropout}
                      onChange={(e) => handleChange('lora_dropout', parseFloat(e.target.value) || 0)}
                      min={0}
                      max={1}
                      step={0.01}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                    <p className="text-xs text-gray-500 mt-1">0 for Unsloth, 0.05 for standard</p>
                  </div>
                </div>
              </div>
            )}

            {/* LR Scheduler Section */}
            {activeSection === 'scheduler' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Learning Rate Scheduler
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {LR_SCHEDULERS.map((scheduler) => (
                      <button
                        key={scheduler.id}
                        onClick={() => handleChange('lr_scheduler_type', scheduler.id)}
                        className={clsx(
                          'p-3 rounded-lg border-2 text-left transition-colors',
                          config.lr_scheduler_type === scheduler.id
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                            : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                        )}
                      >
                        <p className="font-medium text-gray-900 dark:text-gray-100 text-sm">
                          {scheduler.name}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                          {scheduler.description}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>

                {config.lr_scheduler_type === 'cosine_with_restarts' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Number of Cycles
                    </label>
                    <input
                      type="number"
                      value={config.num_cycles || 0.5}
                      onChange={(e) => handleChange('num_cycles', parseFloat(e.target.value) || 0.5)}
                      min={0.1}
                      max={5}
                      step={0.1}
                      className="w-full max-w-xs px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>
                )}
              </div>
            )}

            {/* Advanced Section */}
            {activeSection === 'advanced' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Optimizer
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {OPTIMIZERS.map((opt) => (
                      <button
                        key={opt.id}
                        onClick={() => handleChange('optim', opt.id)}
                        className={clsx(
                          'p-3 rounded-lg border-2 text-left transition-colors',
                          config.optim === opt.id
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                            : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                        )}
                      >
                        <p className="font-medium text-gray-900 dark:text-gray-100 text-sm">
                          {opt.name}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                          {opt.description}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Logging Steps
                    </label>
                    <input
                      type="number"
                      value={config.logging_steps}
                      onChange={(e) => handleChange('logging_steps', parseInt(e.target.value) || 10)}
                      min={1}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Save Steps
                    </label>
                    <input
                      type="number"
                      value={config.save_steps}
                      onChange={(e) => handleChange('save_steps', parseInt(e.target.value) || 100)}
                      min={1}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Weight Decay
                    </label>
                    <input
                      type="number"
                      value={config.weight_decay}
                      onChange={(e) => handleChange('weight_decay', parseFloat(e.target.value) || 0)}
                      min={0}
                      max={1}
                      step={0.001}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  </div>
                  <div className="flex items-center">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.gradient_checkpointing}
                        onChange={(e) => handleChange('gradient_checkpointing', e.target.checked)}
                        className="w-4 h-4 text-primary-600 rounded border-gray-300 focus:ring-primary-500"
                      />
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Gradient Checkpointing
                      </span>
                    </label>
                  </div>
                </div>
              </div>
            )}

            {/* Stats */}
            {datasetSize && datasetSize > 0 && (
              <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-6 text-sm">
                  <div className="flex items-center gap-2">
                    <Timer className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600 dark:text-gray-400">
                      Effective batch: <span className="font-medium text-gray-900 dark:text-gray-100">{effectiveBatchSize}</span>
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-600 dark:text-gray-400">
                      Total steps: <span className="font-medium text-gray-900 dark:text-gray-100">{totalSteps.toLocaleString()}</span>
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
