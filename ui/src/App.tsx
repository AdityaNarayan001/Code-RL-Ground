import { useState, useEffect, useCallback, useRef } from 'react'
import Dashboard from './components/Dashboard'
import { TrainingStatus, PRInfo, WSMessage, TrainingMetrics, AdvancedMetrics, PhaseInfo } from './types'

function App() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [prs, setPrs] = useState<PRInfo[]>([])
  const [metrics, setMetrics] = useState<TrainingMetrics>({ steps: [], episodes: [] })
  const [logs, setLogs] = useState<WSMessage[]>([])
  const [generatingText, setGeneratingText] = useState('')
  const [connected, setConnected] = useState(false)
  const [advancedMetrics, setAdvancedMetrics] = useState<AdvancedMetrics | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [phaseInfo, setPhaseInfo] = useState<PhaseInfo | null>(null)
  const [phaseBanner, setPhaseBanner] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectDelayRef = useRef(1000)
  const pongTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Clear error after 10 seconds
  useEffect(() => {
    if (!errorMessage) return
    const t = setTimeout(() => setErrorMessage(null), 10000)
    return () => clearTimeout(t)
  }, [errorMessage])

  // Clear phase banner after 5 seconds
  useEffect(() => {
    if (!phaseBanner) return
    const t = setTimeout(() => setPhaseBanner(null), 5000)
    return () => clearTimeout(t)
  }, [phaseBanner])

  // Fetch initial data
  const fetchData = useCallback(async () => {
    try {
      const [statusRes, prsRes, metricsRes] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/prs'),
        fetch('/api/metrics'),
      ])

      if (statusRes.ok) {
        const statusData = await statusRes.json()
        setStatus(statusData)
        if (statusData.phase) setPhaseInfo(statusData.phase)
      }
      if (prsRes.ok) setPrs(await prsRes.json())
      if (metricsRes.ok) setMetrics(await metricsRes.json())
    } catch (err) {
      console.error('Failed to fetch data:', err)
      setErrorMessage('Failed to fetch training data from server.')
    }
  }, [])

  // Fetch advanced metrics
  const fetchAdvancedMetrics = useCallback(async () => {
    try {
      const res = await fetch('/api/training/advanced-metrics')
      if (res.ok) {
        setAdvancedMetrics(await res.json())
      }
    } catch {
      // Silently ignore - endpoint may not exist yet
    }
  }, [])

  // Poll advanced metrics every 5 seconds
  useEffect(() => {
    fetchAdvancedMetrics()
    const interval = setInterval(fetchAdvancedMetrics, 5000)
    return () => clearInterval(interval)
  }, [fetchAdvancedMetrics])

  // Setup WebSocket connection with exponential backoff and pong timeout
  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
    let isMounted = true

    const connect = () => {
      if (!isMounted) return

      const wsUrl = `ws://${window.location.host}/ws`
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        if (isMounted) {
          setConnected(true)
          reconnectDelayRef.current = 1000 // Reset backoff on success
          console.log('WebSocket connected')
        }
      }

      ws.onclose = (event) => {
        if (isMounted) {
          setConnected(false)
          // Clear any pending pong timeout
          if (pongTimeoutRef.current) {
            clearTimeout(pongTimeoutRef.current)
            pongTimeoutRef.current = null
          }
          console.log('WebSocket disconnected', event.code)
          // Reconnect with exponential backoff (but not on clean close or unmount)
          if (event.code !== 1000 && isMounted) {
            const delay = reconnectDelayRef.current
            reconnectDelayRef.current = Math.min(delay * 2, 30000)
            reconnectTimeout = setTimeout(connect, delay)
          }
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setErrorMessage('WebSocket connection error. Retrying...')
      }

    ws.onmessage = (event) => {
      try {
        // Handle pong - reset pong timeout
        if (event.data === 'pong') {
          if (pongTimeoutRef.current) {
            clearTimeout(pongTimeoutRef.current)
            pongTimeoutRef.current = null
          }
          return
        }

        const data: WSMessage = JSON.parse(event.data)
        // Add client-side timestamp if server didn't provide one
        if (!data.timestamp) {
          data.timestamp = new Date().toISOString()
        }
        setLogs(prev => [...prev.slice(-999), data])

        // Handle different message types
        switch (data.type) {
          case 'generation_token':
            setGeneratingText(data.full_text || '')
            // Fetch status on first turn to update current PR immediately
            if (data.turn === 1) {
              fetch('/api/status').then(r => r.json()).then(s => { setStatus(s); if (s.phase) setPhaseInfo(s.phase) }).catch(() => {})
            }
            break
          case 'generation_complete':
            setGeneratingText('')
            // Also refresh metrics and status
            fetch('/api/metrics').then(r => r.json()).then(setMetrics).catch(() => {})
            fetch('/api/status').then(r => r.json()).then(s => { setStatus(s); if (s.phase) setPhaseInfo(s.phase) }).catch(() => {})
            break
          case 'step':
          case 'episode':
            // Refresh metrics and status
            fetch('/api/metrics').then(r => r.json()).then(setMetrics).catch(() => {})
            fetch('/api/status').then(r => r.json()).then(s => { setStatus(s); if (s.phase) setPhaseInfo(s.phase) }).catch(() => {})
            break
          case 'pr_solved':
            // Refresh PRs and status
            fetchData()
            break
          case 'training_complete':
          case 'training_error':
            if (data.type === 'training_error') {
              setErrorMessage(`Training error: ${data.message || data.error || 'Unknown error'}`)
            }
            fetchData()
            break
          case 'phase_change': {
            const newPhase: PhaseInfo = {
              current_phase: data.new_phase ?? data.current_phase ?? 0,
              phase_name: data.phase_name ?? data.new_phase_name ?? '',
              advancement_progress: data.advancement_progress ?? {
                recent_rewards: [],
                threshold: 0,
                required: 0,
                met: 0,
                window: 0,
              },
            }
            setPhaseInfo(newPhase)
            const prevName = data.previous_phase_name ?? data.old_phase_name ?? `Phase ${(data.old_phase ?? (newPhase.current_phase - 1))}`
            setPhaseBanner(`Phase ${data.old_phase ?? (newPhase.current_phase - 1)} Complete! Advancing to Phase ${newPhase.current_phase}: ${newPhase.phase_name}`)
            void prevName // consumed above as fallback
            fetchData()
            break
          }
          case 'error':
            setErrorMessage(data.message || data.error || 'An error occurred')
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

    // Ping to keep alive, with pong timeout for reconnection
    const pingInterval = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send('ping')
        // Set pong timeout - if no pong within 10s, reconnect
        if (pongTimeoutRef.current) clearTimeout(pongTimeoutRef.current)
        pongTimeoutRef.current = setTimeout(() => {
          console.warn('Pong timeout - reconnecting')
          if (ws) {
            ws.close()
          }
        }, 10000)
      }
    }, 30000)

    return () => {
      isMounted = false
      if (reconnectTimeout) clearTimeout(reconnectTimeout)
      if (pongTimeoutRef.current) clearTimeout(pongTimeoutRef.current)
      clearInterval(pingInterval)
      if (ws) ws.close(1000)  // Clean close
    }
  }, [fetchData])

  // Initial data fetch only (no polling - we use WebSocket for updates)
  useEffect(() => {
    fetchData()
  }, [fetchData])

  // Training control functions
  const startTraining = async () => {
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (res.ok) {
        fetchData()
      } else {
        const body = await res.text()
        setErrorMessage(`Failed to start training: ${body}`)
      }
    } catch (err) {
      console.error('Failed to start training:', err)
      setErrorMessage('Failed to start training. Is the server running?')
    }
  }

  const startPhasedTraining = async () => {
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phased: true }),
      })
      if (res.ok) {
        fetchData()
      } else {
        const body = await res.text()
        setErrorMessage(`Failed to start phased training: ${body}`)
      }
    } catch (err) {
      console.error('Failed to start phased training:', err)
      setErrorMessage('Failed to start phased training. Is the server running?')
    }
  }

  const stopTraining = async () => {
    try {
      const res = await fetch('/api/training/stop', { method: 'POST' })
      if (res.ok) {
        fetchData()
      } else {
        const body = await res.text()
        setErrorMessage(`Failed to stop training: ${body}`)
      }
    } catch (err) {
      console.error('Failed to stop training:', err)
      setErrorMessage('Failed to stop training. Is the server running?')
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
        onStartPhasedTraining={startPhasedTraining}
        onStopTraining={stopTraining}
        advancedMetrics={advancedMetrics}
        errorMessage={errorMessage}
        phaseInfo={phaseInfo}
        phaseBanner={phaseBanner}
      />
    </div>
  )
}

export default App
