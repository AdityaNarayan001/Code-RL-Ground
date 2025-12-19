import { useEffect, useRef } from 'react'
import { WSMessage } from '../types'

interface LiveGenerationProps {
  text: string
  logs: WSMessage[]
}

function LiveGeneration({ text, logs }: LiveGenerationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  
  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [text, logs])

  // Extract recent generation events for display
  const generationLogs = logs
    .filter(l => l.type === 'generation_complete')
    .slice(-5)

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Model Generation</h2>
        <p className="text-sm text-gray-400">
          {text ? 'Generating...' : 'Waiting for generation'}
        </p>
      </div>
      
      <div 
        ref={containerRef}
        className="flex-1 overflow-auto p-4 font-mono text-sm"
      >
        {text ? (
          <div className="bg-code-bg rounded-lg p-4 whitespace-pre-wrap">
            <span className="text-gray-300">{text}</span>
            <span className="cursor-blink" />
          </div>
        ) : (
          <div className="space-y-4">
            {generationLogs.length === 0 ? (
              <div className="text-gray-500 text-center py-8">
                Model output will appear here during training.
              </div>
            ) : (
              generationLogs.map((log, i) => (
                <div key={i} className="bg-code-bg rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2 text-xs text-gray-400">
                    <span className="px-1.5 py-0.5 bg-gray-700 rounded">
                      {log.pr_id}
                    </span>
                    <span>Turn {log.turn}</span>
                  </div>
                  <pre className="whitespace-pre-wrap text-gray-300 text-sm overflow-auto max-h-96">
                    {log.full_text}
                  </pre>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default LiveGeneration
