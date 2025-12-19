import { useEffect, useRef, useState } from 'react'
import { WSMessage } from '../types'

interface LiveGenerationProps {
  text: string
  logs: WSMessage[]
}

function LiveGeneration({ text, logs }: LiveGenerationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [lastGeneration, setLastGeneration] = useState<WSMessage | null>(null)
  
  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [text, logs])

  // Track latest generation for display
  useEffect(() => {
    const latest = logs
      .filter(l => l.type === 'generation_complete' || l.type === 'generation_token')
      .slice(-1)[0]
    if (latest) {
      setLastGeneration(latest)
    }
  }, [logs])

  // Extract recent generation events for history
  const generationLogs = logs
    .filter(l => l.type === 'generation_complete')
    .slice(-5)

  // Determine what to show
  const isStreaming = !!text
  const displayText = text || lastGeneration?.full_text || ''
  const currentPR = lastGeneration?.pr_id || ''
  const currentEpisode = lastGeneration?.episode || lastGeneration?.turn || 0
  const currentGroup = lastGeneration?.group_idx || 0
  const numTurns = lastGeneration?.turns || 0
  const episodeReward = lastGeneration?.reward
  const episodeSolved = lastGeneration?.solved

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Model Generation</h2>
        <div className="flex items-center gap-3 mt-1">
          {currentPR && (
            <span className="px-2 py-0.5 bg-blue-600 rounded text-xs font-medium">
              {currentPR}
            </span>
          )}
          {currentEpisode > 0 && (
            <span className="text-sm text-gray-400">
              Episode {currentEpisode}{currentGroup > 0 ? ` (${currentGroup}/4)` : ''}
              {numTurns > 0 && ` • ${numTurns} turns`}
            </span>
          )}
          {episodeReward !== undefined && (
            <span className={`text-sm ${episodeSolved ? 'text-green-400' : 'text-yellow-400'}`}>
              R={episodeReward.toFixed(2)}{episodeSolved ? ' ✓' : ''}
            </span>
          )}
          <span className="text-sm text-gray-400">
            {isStreaming ? '● Generating...' : displayText ? '' : 'Waiting...'}
          </span>
        </div>
      </div>
      
      <div 
        ref={containerRef}
        className="flex-1 overflow-auto p-4 font-mono text-sm"
      >
        {displayText ? (
          <div className="bg-gray-800 rounded-lg p-4 whitespace-pre-wrap border border-gray-700">
            <span className="text-gray-300">{displayText}</span>
            {isStreaming && <span className="cursor-blink" />}
          </div>
        ) : (
          <div className="text-gray-500 text-center py-8">
            Model output will appear here during training.
            <br />
            <span className="text-xs">Click "Start Training" to begin.</span>
          </div>
        )}
        
        {/* Show episode history below current generation */}
        {generationLogs.length > 1 && !isStreaming && (
          <div className="mt-6">
            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
              Recent Episodes
            </h3>
            <div className="space-y-3">
              {generationLogs.slice(0, -1).reverse().map((log, i) => (
                <details key={i} className="bg-gray-800/50 rounded-lg border border-gray-700/50">
                  <summary className="px-3 py-2 cursor-pointer text-xs text-gray-400 hover:text-gray-300">
                    <span className="ml-1">
                      {log.pr_id} • Episode {log.turn || log.episode} 
                      {log.reward !== undefined && ` • R=${log.reward.toFixed(2)}`}
                      {log.solved && ' ✓'}
                    </span>
                  </summary>
                  <pre className="px-4 py-2 text-xs text-gray-400 whitespace-pre-wrap max-h-40 overflow-auto">
                    {log.full_text?.slice(0, 500)}{log.full_text?.length > 500 ? '...' : ''}
                  </pre>
                </details>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default LiveGeneration
