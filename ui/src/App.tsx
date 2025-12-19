import { useState, useEffect, useCallback, useRef } from 'react'
import Dashboard from './components/Dashboard'
import { TrainingStatus, PRInfo, WSMessage, TrainingMetrics } from './types'

function App() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [prs, setPrs] = useState<PRInfo[]>([])
  const [metrics, setMetrics] = useState<TrainingMetrics>({ steps: [], episodes: [] })
  const [logs, setLogs] = useState<WSMessage[]>([])
  const [generatingText, setGeneratingText] = useState('')
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  // Fetch initial data
  const fetchData = useCallback(async () => {
    try {
      const [statusRes, prsRes, metricsRes] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/prs'),
        fetch('/api/metrics'),
      ])
      
      if (statusRes.ok) setStatus(await statusRes.json())
      if (prsRes.ok) setPrs(await prsRes.json())
      if (metricsRes.ok) setMetrics(await metricsRes.json())
    } catch (err) {
      console.error('Failed to fetch data:', err)
    }
  }, [])

  // Setup WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout | null = null
    let isMounted = true
    
    const connect = () => {
      if (!isMounted) return
      
      const wsUrl = `ws://${window.location.host}/ws`
      ws = new WebSocket(wsUrl)
      
      ws.onopen = () => {
        if (isMounted) {
          setConnected(true)
          console.log('WebSocket connected')
        }
      }
      
      ws.onclose = (event) => {
        if (isMounted) {
          setConnected(false)
          console.log('WebSocket disconnected', event.code)
          // Reconnect after delay (but not on clean close or unmount)
          if (event.code !== 1000 && isMounted) {
            reconnectTimeout = setTimeout(connect, 3000)
          }
        }
      }
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    
    ws.onmessage = (event) => {
      try {
        // Handle ping/pong
        if (event.data === 'pong') return
        
        const data: WSMessage = JSON.parse(event.data)
        setLogs(prev => [...prev.slice(-999), data])
        
        // Handle different message types
        switch (data.type) {
          case 'generation_token':
            setGeneratingText(data.full_text || '')
            break
          case 'generation_complete':
            setGeneratingText('')
            break
          case 'step':
          case 'episode':
            // Refresh metrics
            fetch('/api/metrics').then(r => r.json()).then(setMetrics).catch(() => {})
            break
          case 'pr_solved':
            // Refresh PRs and status
            fetchData()
            break
          case 'training_complete':
          case 'training_error':
            fetchData()
            break
        }
      } catch (err) {
        // Ignore parse errors for non-JSON messages
        if (event.data !== 'pong') {
          console.error('Failed to parse WS message:', err)
        }
      }
    }
    
    wsRef.current = ws
    }
    
    // Initial connect
    connect()
    
    // Ping to keep alive
    const pingInterval = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send('ping')
      }
    }, 30000)
    
    return () => {
      isMounted = false
      if (reconnectTimeout) clearTimeout(reconnectTimeout)
      clearInterval(pingInterval)
      if (ws) ws.close(1000)  // Clean close
    }
  }, [fetchData])

  // Initial data fetch
  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [fetchData])

  // Training control functions
  const startTraining = async () => {
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (res.ok) fetchData()
    } catch (err) {
      console.error('Failed to start training:', err)
    }
  }

  const stopTraining = async () => {
    try {
      const res = await fetch('/api/training/stop', { method: 'POST' })
      if (res.ok) fetchData()
    } catch (err) {
      console.error('Failed to stop training:', err)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900">
      <Dashboard
        status={status}
        prs={prs}
        metrics={metrics}
        logs={logs}
        generatingText={generatingText}
        connected={connected}
        onStartTraining={startTraining}
        onStopTraining={stopTraining}
      />
    </div>
  )
}

export default App
