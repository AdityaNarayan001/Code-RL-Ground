import { TrendingUp, Target, Layers, Activity, Timer, BarChart3, Gauge } from 'lucide-react'
import { AdvancedMetrics, StepMetric } from '../types'

interface MetricsPanelProps {
  avgReward: number
  solvedPRs: number
  totalPRs: number
  step: number
  advancedMetrics?: AdvancedMetrics | null
  latestSteps?: StepMetric[]
}

function Sparkline({ values, height = 20, width = 80 }: { values: number[]; height?: number; width?: number }) {
  if (values.length === 0) return null
  const max = Math.max(...values, 0.01)
  const min = Math.min(...values, 0)
  const range = max - min || 1
  const barWidth = Math.max(2, Math.floor(width / values.length) - 1)

  return (
    <svg width={width} height={height} className="inline-block align-middle">
      {values.map((v, i) => {
        const barH = ((v - min) / range) * (height - 2)
        return (
          <rect
            key={i}
            x={i * (barWidth + 1)}
            y={height - barH - 1}
            width={barWidth}
            height={Math.max(1, barH)}
            fill="#8b5cf6"
            opacity={0.8}
          />
        )
      })}
    </svg>
  )
}

function MetricsPanel({ avgReward, solvedPRs, totalPRs, step, advancedMetrics, latestSteps }: MetricsPanelProps) {
  const progress = totalPRs > 0 ? (solvedPRs / totalPRs) * 100 : 0

  // Derive gradient variance and step timing from latest steps if available
  const lastStep = latestSteps && latestSteps.length > 0 ? latestSteps[latestSteps.length - 1] : null
  const gradientVariance = advancedMetrics?.gradient_stats?.variance ?? lastStep?.gradient_variance ?? null
  const avgStepMs = advancedMetrics?.step_timing?.avg_ms ?? null
  const lastStepMs = lastStep?.step_duration_ms ?? advancedMetrics?.step_timing?.last_ms ?? null

  // Episode length
  const episodeLengthAvg = advancedMetrics?.episode_length_avg ?? null

  // Reward distribution sparkline
  const rewardDist = advancedMetrics?.reward_distribution ?? []

  return (
    <div className="h-full bg-gray-800 rounded-lg p-4 flex flex-col gap-3 overflow-auto">
      <h3 className="text-sm font-medium text-gray-400">Training Metrics</h3>

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center gap-2 text-gray-400 mb-1">
            <TrendingUp size={14} />
            <span className="text-xs">Avg Reward</span>
          </div>
          <div className="text-xl font-bold text-green-400">
            {avgReward.toFixed(3)}
          </div>
        </div>

        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center gap-2 text-gray-400 mb-1">
            <Target size={14} />
            <span className="text-xs">Solve Rate</span>
          </div>
          <div className="text-xl font-bold text-blue-400">
            {progress.toFixed(1)}%
          </div>
        </div>

        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center gap-2 text-gray-400 mb-1">
            <Layers size={14} />
            <span className="text-xs">PRs Solved</span>
          </div>
          <div className="text-xl font-bold text-purple-400">
            {solvedPRs}/{totalPRs}
          </div>
        </div>

        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center gap-2 text-gray-400 mb-1">
            <Activity size={14} />
            <span className="text-xs">Total Steps</span>
          </div>
          <div className="text-xl font-bold text-orange-400">
            {step.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Extended metrics */}
      <div className="space-y-2 text-xs">
        {gradientVariance !== null && (
          <div className="flex justify-between items-center text-gray-400">
            <span className="flex items-center gap-1"><Gauge size={12} /> Grad Variance</span>
            <span className="text-gray-200 font-mono">{gradientVariance.toExponential(2)}</span>
          </div>
        )}
        {(avgStepMs !== null || lastStepMs !== null) && (
          <div className="flex justify-between items-center text-gray-400">
            <span className="flex items-center gap-1"><Timer size={12} /> Step Timing</span>
            <span className="text-gray-200 font-mono">
              {avgStepMs !== null ? `avg ${(avgStepMs / 1000).toFixed(1)}s` : ''}
              {avgStepMs !== null && lastStepMs !== null ? ' / ' : ''}
              {lastStepMs !== null ? `last ${(lastStepMs / 1000).toFixed(1)}s` : ''}
            </span>
          </div>
        )}
        {episodeLengthAvg !== null && (
          <div className="flex justify-between items-center text-gray-400">
            <span className="flex items-center gap-1"><BarChart3 size={12} /> Avg Ep Length</span>
            <span className="text-gray-200 font-mono">{episodeLengthAvg.toFixed(1)} turns</span>
          </div>
        )}
        {rewardDist.length > 0 && (
          <div className="text-gray-400">
            <div className="flex justify-between items-center mb-1">
              <span className="flex items-center gap-1"><BarChart3 size={12} /> Reward Dist</span>
            </div>
            <Sparkline values={rewardDist} height={24} width={120} />
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div className="mt-auto pt-3 border-t border-gray-700">
        <div className="flex justify-between text-xs text-gray-400 mb-2">
          <span>Progress</span>
          <span>{solvedPRs} of {totalPRs} PRs</span>
        </div>
        <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
    </div>
  )
}

export default MetricsPanel
