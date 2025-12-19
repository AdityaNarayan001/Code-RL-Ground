import { CheckCircle, Circle, Loader, Star } from 'lucide-react'
import { PRInfo } from '../types'
import clsx from 'clsx'

interface PRViewerProps {
  prs: PRInfo[]
  currentPR: string | null
}

function PRViewer({ prs, currentPR }: PRViewerProps) {
  const getDifficultyStars = (difficulty: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <Star
        key={i}
        size={12}
        className={clsx(
          i < difficulty ? 'text-yellow-400 fill-yellow-400' : 'text-gray-600'
        )}
      />
    ))
  }

  const getStatusIcon = (status: PRInfo['status'], isCurrent: boolean) => {
    if (status === 'solved') {
      return <CheckCircle size={16} className="text-green-400" />
    }
    if (isCurrent || status === 'in_progress') {
      return <Loader size={16} className="text-blue-400 animate-spin" />
    }
    return <Circle size={16} className="text-gray-500" />
  }

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">PR Tasks</h2>
        <p className="text-sm text-gray-400">
          {prs.filter(p => p.status === 'solved').length} of {prs.length} completed
        </p>
      </div>
      
      <div className="flex-1 overflow-auto p-2">
        {prs.map((pr) => {
          const isCurrent = pr.pr_id === currentPR
          
          return (
            <div
              key={pr.pr_id}
              className={clsx(
                'p-3 rounded-lg mb-2 transition',
                isCurrent && 'bg-blue-600/20 border border-blue-500/50',
                pr.status === 'solved' && !isCurrent && 'bg-green-600/10 border border-green-500/30',
                pr.status === 'pending' && !isCurrent && 'bg-gray-700/50 hover:bg-gray-700'
              )}
            >
              <div className="flex items-start gap-3">
                {getStatusIcon(pr.status, isCurrent)}
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400 font-mono">
                      {pr.pr_id}
                    </span>
                    {isCurrent && (
                      <span className="text-xs px-1.5 py-0.5 bg-blue-500 text-white rounded">
                        CURRENT
                      </span>
                    )}
                  </div>
                  
                  <h3 className="text-sm font-medium text-white truncate mt-1">
                    {pr.title}
                  </h3>
                  
                  <p className="text-xs text-gray-400 line-clamp-2 mt-1">
                    {pr.description}
                  </p>
                  
                  <div className="flex items-center justify-between mt-2">
                    <div className="flex items-center gap-1">
                      {getDifficultyStars(pr.difficulty)}
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {pr.attempts > 0 && (
                        <span className="text-xs text-gray-500">
                          {pr.attempts} attempts
                        </span>
                      )}
                      {pr.best_reward > 0 && (
                        <span className="text-xs text-gray-400">
                          R={pr.best_reward.toFixed(2)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default PRViewer
