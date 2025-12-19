import { TrendingUp, Target, Layers, Activity } from 'lucide-react'

interface MetricsPanelProps {
  avgReward: number
  solvedPRs: number
  totalPRs: number
  step: number
}

function MetricsPanel({ avgReward, solvedPRs, totalPRs, step }: MetricsPanelProps) {
  const progress = totalPRs > 0 ? (solvedPRs / totalPRs) * 100 : 0

  return (
    <div className="h-full bg-gray-800 rounded-lg p-4 flex flex-col gap-4">
      <h3 className="text-sm font-medium text-gray-400">Training Metrics</h3>
      
      <div className="grid grid-cols-2 gap-4">
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

      {/* Progress bar */}
      <div className="mt-auto">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Progress</span>
          <span>{solvedPRs} of {totalPRs} PRs</span>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
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
