import { Play, Square, Wifi, WifiOff, Clock, Zap, CheckCircle, AlertTriangle } from 'lucide-react'
import { TrainingStatus, PRInfo, TrainingMetrics, WSMessage, AdvancedMetrics } from '../types'
import MetricsPanel from './MetricsPanel'
import RewardChart from './RewardChart'
import LossChart from './LossChart'
import PRViewer from './PRViewer'
import LiveGeneration from './LiveGeneration'
import ToolCallLog from './ToolCallLog'

interface DashboardProps {
  status: TrainingStatus | null
  prs: PRInfo[]
  metrics: TrainingMetrics
  logs: WSMessage[]
  generatingText: string
  connected: boolean
  onStartTraining: () => void
  onStopTraining: () => void
  advancedMetrics?: AdvancedMetrics | null
  errorMessage?: string | null
}

function Dashboard({
  status,
  prs,
  metrics,
  logs,
  generatingText,
  connected,
  onStartTraining,
  onStopTraining,
  advancedMetrics,
  errorMessage,
}: DashboardProps) {
  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  const solvedCount = prs.filter(p => p.status === 'solved').length
  const totalCount = prs.length

  const am = advancedMetrics

  return (
    <div className="flex flex-col h-screen">
      {/* Error banner */}
      {errorMessage && (
        <div className="bg-red-900/80 border-b border-red-700 px-6 py-2 flex items-center gap-2 text-red-200 text-sm">
          <AlertTriangle size={16} />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-white">Code-RL-Ground</h1>
            <span className={`flex items-center gap-1 text-sm ${connected ? 'text-green-400' : 'text-red-400'}`}>
              {connected ? <Wifi size={16} /> : <WifiOff size={16} />}
              {connected ? 'Connected' : 'Disconnected'}
            </span>
            {status?.device && (
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                status.device === 'CUDA' ? 'bg-green-600 text-white' :
                status.device === 'MPS' ? 'bg-blue-600 text-white' :
                'bg-gray-600 text-white'
              }`}>
                {status.device}
              </span>
            )}
          </div>

          <div className="flex items-center gap-4">
            {status && (
              <>
                <div className="flex items-center gap-2 text-gray-400">
                  <Clock size={16} />
                  <span>{formatTime(status.elapsed_time)}</span>
                </div>
                <div className="flex items-center gap-2 text-gray-400">
                  <Zap size={16} />
                  <span>Step {status.current_step}</span>
                </div>
                <div className="flex items-center gap-2 text-gray-400">
                  <CheckCircle size={16} />
                  <span>{solvedCount}/{totalCount} PRs</span>
                </div>
              </>
            )}

            {status?.is_running ? (
              <button
                onClick={onStopTraining}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition"
              >
                <Square size={16} />
                Stop Training
              </button>
            ) : (
              <button
                onClick={onStartTraining}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition"
              >
                <Play size={16} />
                Start Training
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Advanced metrics bar */}
      {am && (
        <div className="bg-gray-800/50 border-b border-gray-700 px-6 py-2 flex items-center gap-6 text-xs text-gray-400 overflow-x-auto">
          {am.memory_usage && am.memory_usage.total_mb > 0 && (
            <div className="flex items-center gap-2 shrink-0">
              <span>Memory:</span>
              <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all"
                  style={{ width: `${Math.min(100, (am.memory_usage.used_mb / am.memory_usage.total_mb) * 100)}%` }}
                />
              </div>
              <span className="text-gray-300">
                {(am.memory_usage.used_mb / 1024).toFixed(1)}/{(am.memory_usage.total_mb / 1024).toFixed(1)} GB
              </span>
            </div>
          )}
          {am.curriculum_progress && am.curriculum_progress.total > 0 && (
            <div className="shrink-0">
              <span>Curriculum: </span>
              <span className="text-gray-300">
                PR {am.curriculum_progress.current}/{am.curriculum_progress.total}
                {am.curriculum_progress.current_pr_id && ` - ${am.curriculum_progress.current_pr_id}`}
              </span>
            </div>
          )}
          {am.reward_std !== null && (
            <div className="shrink-0">
              <span>Reward Std: </span>
              <span className="text-gray-300">{am.reward_std.toFixed(4)}</span>
            </div>
          )}
          {am.step_timing && am.step_timing.avg_ms > 0 && (
            <div className="shrink-0">
              <span>Avg Step: </span>
              <span className="text-gray-300">{(am.step_timing.avg_ms / 1000).toFixed(1)}s</span>
            </div>
          )}
          {am.policy_entropy !== null && (
            <div className="shrink-0">
              <span>Policy Entropy: </span>
              <span className="text-gray-300">{am.policy_entropy.toFixed(4)}</span>
            </div>
          )}
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar - PRs */}
        <aside className="w-80 bg-gray-800 border-r border-gray-700 overflow-auto">
          <PRViewer prs={prs} currentPR={status?.current_pr || null} />
        </aside>

        {/* Main area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Charts */}
          <div className="h-80 border-b border-gray-700 p-4 flex gap-4 shrink-0">
            <div className="flex-1 bg-gray-800 rounded-lg p-4 min-w-0">
              <h3 className="text-sm font-medium text-gray-400 mb-2">Reward Over Episodes</h3>
              <div className="h-[calc(100%-28px)]">
                <RewardChart data={metrics.episodes} currentStep={status?.current_step} />
              </div>
            </div>
            <div className="flex-1 bg-gray-800 rounded-lg p-4 min-w-0">
              <h3 className="text-sm font-medium text-gray-400 mb-2">Loss Over Steps</h3>
              <div className="h-[calc(100%-28px)]">
                <LossChart data={metrics.steps} currentStep={status?.current_step} />
              </div>
            </div>
            <div className="w-72 shrink-0">
              <MetricsPanel
                avgReward={status?.avg_reward || 0}
                solvedPRs={solvedCount}
                totalPRs={totalCount}
                step={status?.current_step || 0}
                advancedMetrics={advancedMetrics}
                latestSteps={metrics.steps.slice(-10)}
              />
            </div>
          </div>

          {/* Generation + Logs */}
          <div className="flex-1 flex overflow-hidden min-h-0">
            <div className="flex-1 border-r border-gray-700 overflow-hidden min-w-0">
              <LiveGeneration text={generatingText} logs={logs} />
            </div>
            <div className="w-96 shrink-0 overflow-hidden">
              <ToolCallLog logs={logs} />
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

export default Dashboard
