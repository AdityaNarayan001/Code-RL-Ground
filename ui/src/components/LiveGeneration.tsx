import { useEffect, useRef, useState } from 'react'
import { WSMessage } from '../types'

interface LiveGenerationProps {
  text: string
  logs: WSMessage[]
}

function LiveGeneration({ text, logs }: LiveGenerationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [lastComplete, setLastComplete] = useState<WSMessage | null>(null)
  const [lastToken, setLastToken] = useState<WSMessage | null>(null)
  
  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [text, logs])

  // Track latest generation events separately - scan backwards for efficiency
  useEffect(() => {
    // Find latest generation_complete
    for (let i = logs.length - 1; i >= 0; i--) {
      if (logs[i].type === 'generation_complete') {
        setLastComplete(logs[i])
        break
      }
    }
    
    // Find latest generation_token
    for (let i = logs.length - 1; i >= 0; i--) {
      if (logs[i].type === 'generation_token') {
        setLastToken(logs[i])
        break
      }
    }
  }, [logs])

  // Extract recent generation events for history
  const generationLogs = logs
    .filter(l => l.type === 'generation_complete')
    .slice(-5)

  // Determine what to show
  const isStreaming = !!text
  const displayText = text || lastToken?.full_text || lastComplete?.full_text || ''
  const currentPR = (isStreaming ? lastToken?.pr_id : lastComplete?.pr_id) || lastToken?.pr_id || lastComplete?.pr_id || ''
  
  // During streaming, show turn and group from the latest token
  // After complete, show episode info from the latest complete
  const currentTurn = lastToken?.turn || 0
  const currentEpisode = lastComplete?.episode || 0
  const currentGroup = isStreaming 
    ? (lastToken?.group_idx || lastComplete?.group_idx || 0)
    : (lastComplete?.group_idx || 0)
  const numTurns = lastComplete?.turns || 0
  const episodeReward = lastComplete?.reward
  const episodeSolved = lastComplete?.solved

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
          {isStreaming && currentTurn > 0 && (
            <span className="text-sm text-yellow-400">
              Turn {currentTurn}/10 • Episode {currentGroup}/4
            </span>
          )}
          {!isStreaming && currentEpisode > 0 && (
            <span className="text-sm text-gray-400">
              Episode {currentEpisode} (Group {currentGroup}/4)
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
                    {log.full_text?.slice(0, 500)}{(log.full_text?.length || 0) > 500 ? '...' : ''}
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
