import { useEffect, useRef } from 'react'
import { Terminal, FileCode, Play, Search, Send, CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import { WSMessage } from '../types'
import clsx from 'clsx'

interface ToolCallLogProps {
  logs: WSMessage[]
}

const toolIcons: Record<string, React.ReactNode> = {
  read_file: <FileCode size={14} />,
  write_file: <FileCode size={14} className="text-yellow-400" />,
  edit_file: <FileCode size={14} className="text-blue-400" />,
  run_python: <Terminal size={14} className="text-green-400" />,
  search_code: <Search size={14} className="text-purple-400" />,
  submit: <Send size={14} className="text-orange-400" />,
  list_directory: <FileCode size={14} className="text-gray-400" />,
}

function ToolCallLog({ logs }: ToolCallLogProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs])

  // Filter for tool-related and important events
  const relevantLogs = logs.filter(l => 
    ['tool_call', 'tool_result', 'episode', 'pr_solved', 'error', 'checkpoint', 'info', 'training_error', 'training_complete'].includes(l.type)
  ).slice(-100)

  return (
    <div className="h-full flex flex-col bg-gray-800">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Activity Log</h2>
        <p className="text-sm text-gray-400">Tool calls & events</p>
      </div>
      
      <div 
        ref={containerRef}
        className="flex-1 overflow-auto p-2 text-xs"
      >
        {relevantLogs.length === 0 ? (
          <div className="text-gray-500 text-center py-8">
            Activity will appear here.
          </div>
        ) : (
          <div className="space-y-1">
            {relevantLogs.map((log, i) => (
              <LogEntry key={i} log={log} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function LogEntry({ log }: { log: WSMessage }) {
  const getIcon = () => {
    switch (log.type) {
      case 'tool_call':
        return toolIcons[log.tool] || <Play size={14} />
      case 'tool_result':
        return log.success ? (
          <CheckCircle size={14} className="text-green-400" />
        ) : (
          <XCircle size={14} className="text-red-400" />
        )
      case 'episode':
        return log.solved ? (
          <CheckCircle size={14} className="text-green-400" />
        ) : (
          <AlertCircle size={14} className="text-yellow-400" />
        )
      case 'pr_solved':
        return <CheckCircle size={14} className="text-green-400" />
      case 'checkpoint':
        return <FileCode size={14} className="text-blue-400" />
      case 'error':
      case 'training_error':
        return <XCircle size={14} className="text-red-400" />
      case 'info':
        return <AlertCircle size={14} className="text-blue-400" />
      case 'training_complete':
        return <CheckCircle size={14} className="text-green-400" />
      default:
        return <Play size={14} />
    }
  }

  const getMessage = () => {
    switch (log.type) {
      case 'tool_call':
        return (
          <span>
            <span className="text-blue-400">{log.tool}</span>
            {log.args && (
              <span className="text-gray-500 ml-1 truncate">
                {JSON.stringify(log.args).slice(0, 50)}
              </span>
            )}
          </span>
        )
      case 'tool_result':
        return (
          <span className={log.success ? 'text-green-400' : 'text-red-400'}>
            {log.success ? 'Success' : 'Failed'}
            {log.output && (
              <span className="text-gray-500 ml-1">
                {String(log.output).slice(0, 50)}
              </span>
            )}
          </span>
        )
      case 'episode':
        return (
          <span>
            Episode {log.episode}: {log.pr_id} - 
            <span className={log.solved ? 'text-green-400' : 'text-yellow-400'}>
              {' '}{log.solved ? 'Solved!' : `R=${log.reward?.toFixed(2)}`}
            </span>
          </span>
        )
      case 'pr_solved':
        return (
          <span className="text-green-400 font-medium">
            🎉 {log.pr_id} SOLVED!
          </span>
        )
      case 'checkpoint':
        return (
          <span className="text-blue-400">
            Checkpoint saved: step {log.step}
          </span>
        )
      case 'error':
      case 'training_error':
        return (
          <span className="text-red-400">
            {log.message || log.error}
          </span>
        )
      case 'info':
        return (
          <span className="text-blue-400">
            {log.message}
          </span>
        )
      case 'training_complete':
        return (
          <span className="text-green-400 font-medium">
            🎉 Training complete!
          </span>
        )
      default:
        return log.message || log.type
    }
  }

  return (
    <div className={clsx(
      'flex items-center gap-2 p-2 rounded',
      log.type === 'pr_solved' && 'bg-green-900/30',
      log.type === 'training_complete' && 'bg-green-900/30',
      (log.type === 'error' || log.type === 'training_error') && 'bg-red-900/30',
      log.type === 'checkpoint' && 'bg-blue-900/20',
      log.type === 'info' && 'bg-blue-900/10'
    )}>
      {getIcon()}
      <div className="flex-1 truncate text-gray-300">
        {getMessage()}
      </div>
    </div>
  )
}

export default ToolCallLog
