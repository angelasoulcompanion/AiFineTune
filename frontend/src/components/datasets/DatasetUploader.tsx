import { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, AlertCircle, X } from 'lucide-react';
import { useUploadDataset } from '../../hooks/useDatasets';
import clsx from 'clsx';

const ACCEPTED_FORMATS = ['.jsonl', '.json', '.csv', '.parquet'];
const MAX_SIZE_MB = 500;

const datasetTypes = [
  { value: 'sft', label: 'SFT (Instruction-Output)', description: 'For supervised fine-tuning' },
  { value: 'chat', label: 'Chat (Messages)', description: 'ChatML format with messages array' },
  { value: 'dpo', label: 'DPO (Preferences)', description: 'prompt, chosen, rejected format' },
  { value: 'orpo', label: 'ORPO (Preferences)', description: 'Same as DPO format' },
];

export default function DatasetUploader() {
  const navigate = useNavigate();
  const uploadMutation = useUploadDataset();

  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [datasetType, setDatasetType] = useState('sft');
  const [tags, setTags] = useState('');
  const [error, setError] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const droppedFile = e.dataTransfer.files[0];
    handleFileSelect(droppedFile);
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    setError('');

    // Check extension
    const ext = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
    if (!ACCEPTED_FORMATS.includes(ext)) {
      setError(`Invalid file format. Accepted: ${ACCEPTED_FORMATS.join(', ')}`);
      return;
    }

    // Check size
    if (selectedFile.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`File too large. Maximum size: ${MAX_SIZE_MB} MB`);
      return;
    }

    setFile(selectedFile);

    // Auto-fill name from filename
    if (!name) {
      const baseName = selectedFile.name.replace(/\.[^/.]+$/, '');
      setName(baseName);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!file) {
      setError('Please select a file');
      return;
    }

    if (!name.trim()) {
      setError('Please enter a name');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name.trim());
    formData.append('dataset_type', datasetType);
    if (description) formData.append('description', description);
    if (tags) formData.append('tags', tags);

    try {
      const dataset = await uploadMutation.mutateAsync(formData);
      navigate(`/datasets/${dataset.dataset_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload dataset');
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Upload Dataset</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Upload a dataset for fine-tuning
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* File drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          className={clsx(
            'relative border-2 border-dashed rounded-xl p-8 text-center transition-colors',
            isDragOver
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400',
            file && 'border-green-500 bg-green-50 dark:bg-green-900/20'
          )}
        >
          {file ? (
            <div className="flex items-center justify-center gap-4">
              <FileText className="w-10 h-10 text-green-500" />
              <div className="text-left">
                <p className="font-medium text-gray-900 dark:text-gray-100">{file.name}</p>
                <p className="text-sm text-gray-500">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
              <button
                type="button"
                onClick={() => setFile(null)}
                className="p-1 text-gray-400 hover:text-red-500"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          ) : (
            <>
              <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <p className="text-gray-600 dark:text-gray-400 mb-2">
                Drag and drop your file here, or click to browse
              </p>
              <p className="text-sm text-gray-500">
                Supported formats: {ACCEPTED_FORMATS.join(', ')} (max {MAX_SIZE_MB} MB)
              </p>
              <input
                type="file"
                accept={ACCEPTED_FORMATS.join(',')}
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </>
          )}
        </div>

        {/* Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Dataset Name *
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            placeholder="My Training Dataset"
          />
        </div>

        {/* Description */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            placeholder="Optional description..."
          />
        </div>

        {/* Dataset Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Dataset Type *
          </label>
          <div className="grid grid-cols-2 gap-3">
            {datasetTypes.map((type) => (
              <label
                key={type.value}
                className={clsx(
                  'flex flex-col p-4 border rounded-lg cursor-pointer transition-colors',
                  datasetType === type.value
                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                )}
              >
                <input
                  type="radio"
                  name="datasetType"
                  value={type.value}
                  checked={datasetType === type.value}
                  onChange={(e) => setDatasetType(e.target.value)}
                  className="sr-only"
                />
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {type.label}
                </span>
                <span className="text-sm text-gray-500">{type.description}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Tags */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Tags
          </label>
          <input
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            placeholder="comma, separated, tags"
          />
        </div>

        {/* Submit */}
        <div className="flex gap-4">
          <button
            type="button"
            onClick={() => navigate('/datasets')}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={uploadMutation.isPending || !file}
            className="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-400 text-white rounded-lg transition-colors"
          >
            {uploadMutation.isPending ? 'Uploading...' : 'Upload Dataset'}
          </button>
        </div>
      </form>
    </div>
  );
}
