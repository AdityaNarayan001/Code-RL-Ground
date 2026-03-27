import { TrendingUp, Target, Layers, Activity, Timer, BarChart3, Gauge, GitBranch } from 'lucide-react'
import { AdvancedMetrics, StepMetric, PhaseInfo } from '../types'

interface MetricsPanelProps {
  avgReward: number
  solvedPRs: number
  totalPRs: number
  step: number
  advancedMetrics?: AdvancedMetrics | null
  latestSteps?: StepMetric[]
  phaseInfo?: PhaseInfo | null
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

function MetricsPanel({ avgReward, solvedPRs, totalPRs, step, advancedMetrics, latestSteps, phaseInfo }: MetricsPanelProps) {
  const progress = totalPRs > 0 ? (solvedPRs / totalPRs) * 100 : 0

  // Derive gradient variance and step timing from latest steps if available
  const lastStep = latestSteps && latestSteps.length > 0 ? latestSteps[latestSteps.length - 1] : null
  const gradientMean = advancedMetrics?.gradient_stats?.mean ?? null
  const avgStepSec = advancedMetrics?.step_timing?.avg_seconds ?? null
  const lastStepMs = lastStep?.step_duration_ms ?? null

  // Episode length
  const episodeLengthAvg = advancedMetrics?.episode_length_distribution?.avg_turns ?? null

  // Reward distribution sparkline (server sends object, extract values)
  const rewardDistObj = advancedMetrics?.reward_distribution ?? {}
  const rewardDist = Object.values(rewardDistObj)

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
        {gradientMean !== null && (
          <div className="flex justify-between items-center text-gray-400">
            <span className="flex items-center gap-1"><Gauge size={12} /> Grad Mean</span>
            <span className="text-gray-200 font-mono">{gradientMean.toExponential(2)}</span>
          </div>
        )}
        {(avgStepSec !== null || lastStepMs !== null) && (
          <div className="flex justify-between items-center text-gray-400">
            <span className="flex items-center gap-1"><Timer size={12} /> Step Timing</span>
            <span className="text-gray-200 font-mono">
              {avgStepSec !== null ? `avg ${avgStepSec.toFixed(1)}s` : ''}
              {avgStepSec !== null && lastStepMs !== null ? ' / ' : ''}
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

      {/* Phase advancement metrics */}
      {phaseInfo && phaseInfo.current_phase > 0 && (
        <div className="space-y-2 text-xs border-t border-gray-700 pt-3">
          <div className="flex justify-between items-center text-gray-400">
            <span className="flex items-center gap-1"><GitBranch size={12} /> Phase</span>
            <span className="text-gray-200 font-mono">{phaseInfo.current_phase}: {phaseInfo.phase_name}</span>
          </div>
          {phaseInfo.advancement_progress && (
            <>
              <div className="flex justify-between items-center text-gray-400">
                <span className="flex items-center gap-1"><Target size={12} /> Advancement</span>
                <span className="text-gray-200 font-mono">
                  {phaseInfo.advancement_progress.met}/{phaseInfo.advancement_progress.required} above {phaseInfo.advancement_progress.threshold.toFixed(2)}
                </span>
              </div>
              {phaseInfo.advancement_progress.recent_rewards.length > 0 && (
                <div className="flex items-center gap-1 mt-1">
                  {phaseInfo.advancement_progress.recent_rewards.map((r, i) => (
                    <div
                      key={i}
                      className={`w-2.5 h-2.5 rounded-full ${
                        r >= phaseInfo.advancement_progress.threshold
                          ? 'bg-green-400'
                          : 'bg-red-400'
                      }`}
                      title={`${r.toFixed(3)}`}
                    />
                  ))}
                  {Array.from({ length: Math.max(0, phaseInfo.advancement_progress.window - phaseInfo.advancement_progress.recent_rewards.length) }).map((_, i) => (
                    <div key={`e-${i}`} className="w-2.5 h-2.5 rounded-full bg-gray-600" />
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}

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
