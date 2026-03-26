export interface PhaseAdvancementProgress {
  recent_rewards: number[]
  threshold: number
  required: number
  met: number
  window: number
}

export interface PhaseInfo {
  current_phase: number       // 0=legacy, 1-5
  phase_name: string          // "Code Completion", "Tool Format", etc.
  advancement_progress: PhaseAdvancementProgress
}

export interface CheckpointInfo {
  has_checkpoints: boolean
  completed_phases: number[]
  resume_phase: number
  latest_checkpoint: string | null
}

export interface TrainingStatus {
  is_running: boolean
  current_step: number
  current_episode: number
  current_pr: string | null
  solved_prs: string[]
  avg_reward: number
  elapsed_time: number
  device: string
  phase?: PhaseInfo
  checkpoints?: CheckpointInfo
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
  pg_loss?: number
  kl_loss?: number
  grad_norm?: number
  clip_frac?: number
  max_reward?: number
  entropy?: number
  gradient_variance?: number
  step_duration_ms?: number
}

export interface EpisodeMetric {
  episode: number
  reward: number
  pr_id: string
  solved: boolean
  duration_ms?: number
  turn_count?: number
}

export interface TrainingMetrics {
  steps: StepMetric[]
  episodes: EpisodeMetric[]
}

export interface AdvancedMetrics {
  policy_entropy: number | null
  reward_std: number | null
  reward_distribution: Record<string, number>
  curriculum_progress: { solved_prs: string[]; current_pr: string | null; attempts_per_pr: Record<string, number> }
  memory_usage: { system_total_mb: number; system_used_mb: number; system_percent: number; gpu_allocated_mb?: number; gpu_reserved_mb?: number }
  step_timing: { avg_seconds?: number; min_seconds?: number; max_seconds?: number }
  gradient_stats: { mean?: number; max?: number; min?: number }
  episode_length_distribution: { avg_turns?: number; min_turns?: number; max_turns?: number }
}

export interface WSMessage {
  type: string
  timestamp?: string
  pr_id?: string
  episode?: number
  turn?: number
  group_idx?: number
  group_size?: number
  max_turns?: number
  turns?: number
  reward?: number
  solved?: boolean
  full_text?: string
  message?: string
  [key: string]: any
}

export interface LogEntry {
  type: string
  timestamp?: string
  message?: string
  [key: string]: any
}
