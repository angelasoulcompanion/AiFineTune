/**
 * Training Loss Visualization Chart
 */
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from 'recharts';

interface LossDataPoint {
  step: number;
  loss: number;
}

interface LRDataPoint {
  step: number;
  lr: number;
}

interface EpochLoss {
  epoch: number;
  avg_loss: number;
}

interface MemoryDataPoint {
  step: number;
  memory_gb: number;
}

interface LossChartProps {
  lossHistory: LossDataPoint[];
  lrHistory?: LRDataPoint[];
  epochLosses?: EpochLoss[];
  memoryHistory?: MemoryDataPoint[];
  height?: number;
}

export default function LossChart({
  lossHistory,
  lrHistory,
  epochLosses,
  memoryHistory,
  height = 300,
}: LossChartProps) {
  if (!lossHistory || lossHistory.length === 0) {
    return (
      <div className="flex items-center justify-center h-[300px] text-gray-500 dark:text-gray-400">
        No training data yet
      </div>
    );
  }

  // Combine data for chart
  const chartData = lossHistory.map((point) => {
    const lrPoint = lrHistory?.find((lr) => lr.step === point.step);
    const memPoint = memoryHistory?.find((m) => m.step === point.step);
    return {
      step: point.step,
      loss: point.loss,
      lr: lrPoint?.lr,
      memory: memPoint?.memory_gb,
    };
  });

  // Sample data if too many points (for performance)
  const sampleRate = Math.ceil(chartData.length / 100);
  const displayData =
    chartData.length > 100
      ? chartData.filter((_, index) => index % sampleRate === 0)
      : chartData;

  // Format tooltip values
  const formatLoss = (value: number) => value?.toFixed(4) ?? '-';
  const formatLR = (value: number) => value?.toExponential(2) ?? '-';
  const formatMemory = (value: number) => `${value?.toFixed(1)} GB`;

  return (
    <div className="space-y-6">
      {/* Loss Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Training Loss
        </h4>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={displayData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis
              dataKey="step"
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={(value) => `${value}`}
            />
            <YAxis
              yAxisId="loss"
              stroke="#9CA3AF"
              fontSize={12}
              tickFormatter={formatLoss}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '8px',
              }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value, name) => {
                if (name === 'loss' && typeof value === 'number') return [formatLoss(value), 'Loss'];
                return [value, name];
              }}
              labelFormatter={(step) => `Step ${step}`}
            />
            <Legend />
            <Area
              yAxisId="loss"
              type="monotone"
              dataKey="loss"
              stroke="#3B82F6"
              fill="#3B82F6"
              fillOpacity={0.1}
              strokeWidth={2}
              dot={false}
              name="Loss"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Learning Rate Chart (if available) */}
      {lrHistory && lrHistory.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Learning Rate Schedule
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="step" stroke="#9CA3AF" fontSize={12} />
              <YAxis
                stroke="#9CA3AF"
                fontSize={12}
                tickFormatter={formatLR}
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
                formatter={(value) => typeof value === 'number' ? [formatLR(value), 'LR'] : [value, 'LR']}
                labelFormatter={(step) => `Step ${step}`}
              />
              <Line
                type="monotone"
                dataKey="lr"
                stroke="#10B981"
                strokeWidth={2}
                dot={false}
                name="Learning Rate"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Memory Usage Chart (if available) */}
      {memoryHistory && memoryHistory.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Memory Usage
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="step" stroke="#9CA3AF" fontSize={12} />
              <YAxis
                stroke="#9CA3AF"
                fontSize={12}
                tickFormatter={(v) => `${v}GB`}
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
                formatter={(value) => typeof value === 'number' ? [formatMemory(value), 'Memory'] : [value, 'Memory']}
                labelFormatter={(step) => `Step ${step}`}
              />
              <Line
                type="monotone"
                dataKey="memory"
                stroke="#F59E0B"
                strokeWidth={2}
                dot={false}
                name="Memory (GB)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Epoch Summary (if available) */}
      {epochLosses && epochLosses.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Average Loss per Epoch
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {epochLosses.map((epoch) => (
              <div
                key={epoch.epoch}
                className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 text-center"
              >
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Epoch {epoch.epoch}
                </p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {epoch.avg_loss.toFixed(4)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
