import { useMemo } from 'react'
import { Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, ReferenceLine } from 'recharts'
import { EpisodeMetric, PhaseInfo, WSMessage } from '../types'

interface RewardChartProps {
  data: EpisodeMetric[]
  currentStep?: number
  phaseInfo?: PhaseInfo | null
  logs?: WSMessage[]
}

function RewardChart({ data, currentStep: _currentStep, phaseInfo: _phaseInfo, logs }: RewardChartProps) {
  // Calculate moving average in a single O(n) pass with a running sum
  const chartData = useMemo(() => {
    const windowSize = 10
    let runningSum = 0
    return data.map((point, i) => {
      runningSum += point.reward
      if (i >= windowSize) {
        runningSum -= data[i - windowSize].reward
      }
      const windowLength = Math.min(i + 1, windowSize)

      return {
        episode: point.episode,
        reward: point.reward,
        avgReward: runningSum / windowLength,
        solved: point.solved ? 1 : 0,
      }
    })
  }, [data])

  // Detect phase transition episodes from logs
  const phaseTransitions = useMemo(() => {
    if (!logs) return []
    const transitions: { episode: number; phase: number; phaseName: string }[] = []
    for (const log of logs) {
      if (log.type === 'phase_change') {
        const ep = log.episode ?? log.at_episode ?? 0
        transitions.push({
          episode: ep,
          phase: log.new_phase ?? log.current_phase ?? 0,
          phaseName: log.phase_name ?? log.new_phase_name ?? '',
        })
      }
    }
    return transitions
  }, [logs])

  if (chartData.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        No data yet. Start training to see metrics.
      </div>
    )
  }

  // Compute Y domain from data with padding
  const allValues = chartData.flatMap(d => [d.reward, d.avgReward])
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)
  const padding = Math.max((maxVal - minVal) * 0.1, 0.05)
  const yDomain: [number, number] = [
    Math.max(0, Math.floor((minVal - padding) * 20) / 20),
    Math.ceil((maxVal + padding) * 20) / 20,
  ]

  // Find current episode (latest)
  const currentEpisode = chartData.length > 0 ? chartData[chartData.length - 1].episode : 0

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <defs>
          <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="episode"
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af', fontSize: 10 }}
        />
        <YAxis
          domain={yDomain}
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af', fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '0.5rem',
          }}
          labelStyle={{ color: '#9ca3af' }}
          formatter={(value: number) => value.toFixed(3)}
          labelFormatter={(label: number) => {
            const transition = phaseTransitions.find(t => t.episode === label)
            if (transition) {
              return `Episode ${label} (Phase ${transition.phase}: ${transition.phaseName})`
            }
            return `Episode ${label}`
          }}
        />
        {/* Phase transition lines */}
        {phaseTransitions.map((t, i) => (
          <ReferenceLine
            key={`phase-${i}`}
            x={t.episode}
            stroke="#a855f7"
            strokeWidth={1.5}
            strokeDasharray="6 3"
            label={{
              value: `P${t.phase}`,
              position: 'top',
              fill: '#a855f7',
              fontSize: 10,
            }}
          />
        ))}
        {/* Current episode indicator */}
        {currentEpisode > 0 && (
          <ReferenceLine
            x={currentEpisode}
            stroke="#f59e0b"
            strokeWidth={2}
            strokeDasharray="4 4"
          />
        )}
        <Area
          type="monotone"
          dataKey="reward"
          stroke="#10b981"
          fill="url(#rewardGradient)"
          strokeWidth={1}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="avgReward"
          stroke="#8b5cf6"
          strokeWidth={2}
          dot={false}
          name="Moving Avg"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export default RewardChart
