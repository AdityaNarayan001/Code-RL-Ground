import { Play, Square, Wifi, WifiOff, Clock, Zap, CheckCircle } from 'lucide-react'
import { TrainingStatus, PRInfo, TrainingMetrics, WSMessage } from '../types'
import MetricsPanel from './MetricsPanel'
import RewardChart from './RewardChart'
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
}: DashboardProps) {
  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  const solvedCount = prs.filter(p => p.status === 'solved').length
  const totalCount = prs.length

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold text-white">Code-RL-Ground</h1>
            <span className={`flex items-center gap-1 text-sm ${connected ? 'text-green-400' : 'text-red-400'}`}>
              {connected ? <Wifi size={16} /> : <WifiOff size={16} />}
              {connected ? 'Connected' : 'Disconnected'}
            </span>
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

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar - PRs */}
        <aside className="w-80 bg-gray-800 border-r border-gray-700 overflow-auto">
          <PRViewer prs={prs} currentPR={status?.current_pr || null} />
        </aside>

        {/* Main area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Charts */}
          <div className="h-72 border-b border-gray-700 p-4 flex gap-6 shrink-0">
            <div className="flex-1 bg-gray-800 rounded-lg p-4 min-w-0">
              <h3 className="text-sm font-medium text-gray-400 mb-2">Reward Over Time</h3>
              <RewardChart data={metrics.episodes} />
            </div>
            <div className="w-80 shrink-0">
              <MetricsPanel 
                avgReward={status?.avg_reward || 0}
                solvedPRs={solvedCount}
                totalPRs={totalCount}
                step={status?.current_step || 0}
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
