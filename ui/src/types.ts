export interface TrainingStatus {
  is_running: boolean
  current_step: number
  current_episode: number
  current_pr: string | null
  solved_prs: string[]
  avg_reward: number
  elapsed_time: number
}

export interface PRInfo {
  pr_id: string
  title: string
  description: string
  difficulty: number
  status: 'solved' | 'in_progress' | 'pending'
  best_reward: number
  attempts: number
}

export interface StepMetric {
  step: number
  avg_reward: number
  solve_rate: number
  loss: number
}

export interface EpisodeMetric {
  episode: number
  reward: number
  pr_id: string
  solved: boolean
}

export interface TrainingMetrics {
  steps: StepMetric[]
  episodes: EpisodeMetric[]
}

export interface WSMessage {
  type: string
  [key: string]: any
}

export interface LogEntry {
  type: string
  timestamp?: string
  message?: string
  [key: string]: any
}
