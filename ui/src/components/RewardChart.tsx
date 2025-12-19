import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { EpisodeMetric } from '../types'

interface RewardChartProps {
  data: EpisodeMetric[]
}

function RewardChart({ data }: RewardChartProps) {
  // Calculate moving average
  const windowSize = 10
  const chartData = data.map((point, i) => {
    const start = Math.max(0, i - windowSize + 1)
    const window = data.slice(start, i + 1)
    const avg = window.reduce((sum, p) => sum + p.reward, 0) / window.length
    
    return {
      episode: point.episode,
      reward: point.reward,
      avgReward: avg,
      solved: point.solved ? 1 : 0,
    }
  })

  if (chartData.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        No data yet. Start training to see metrics.
      </div>
    )
  }

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
          domain={[0, 1]}
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
        />
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
