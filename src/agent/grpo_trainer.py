"""GRPO (Group Relative Policy Optimization) trainer - TinyZero-inspired implementation.

Based on veRL's GRPO implementation:
1. Token-level log probabilities (not sequence sums)
2. Group-relative advantage normalization  
3. Masked loss computation on response tokens only
4. Low-variance KL penalty
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import json
import shutil
import traceback

from ..utils.config import Config
from ..utils.logging import TrainingLogger, get_logger
from ..utils.metrics import ExperimentLogger, create_experiment_logger
from ..environment import CodeEnv
from ..data import PRTask
from ..rewards.format_reward import FormatRewardScorer, FormatRewardConfig
from .policy import LLMPolicy

logger = get_logger(__name__)


@dataclass
class TurnData:
    """Data from a single turn within a multi-turn episode."""
    prompt_ids: torch.Tensor     # (prompt_len,) - full prompt for this turn
    response_ids: torch.Tensor   # (response_len,) - response tokens
    old_log_probs: torch.Tensor  # (response_len,) - per-token log probs from generation


@dataclass
class RolloutData:
    """Data from a single rollout (multi-turn episode)."""
    turns: List[TurnData]             # Per-turn prompt/response pairs
    response_mask: torch.Tensor       # (total_response_len,) - 1 for valid tokens
    reward: float                     # shaped reward (task + format warm-up), used for advantages
    solved: bool
    group_id: str                     # to group responses from same prompt
    response_text: str                # for display
    advantage: float = 0.0
    task_reward: float = 0.0          # unshaped task reward, used for solve/progression checks


@dataclass
class TrainingStats:
    """Track training statistics."""
    total_steps: int = 0
    total_episodes: int = 0
    current_pr_idx: int = 0
    current_pr_id: str = ""
    consecutive_solves: int = 0
    best_rewards: Dict[str, float] = field(default_factory=dict)
    solved_prs: List[str] = field(default_factory=list)
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    attempts_per_pr: Dict[str, int] = field(default_factory=dict)  # Track attempts per PR
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'current_pr_idx': self.current_pr_idx,
            'current_pr_id': self.current_pr_id,
            'consecutive_solves': self.consecutive_solves,
            'best_rewards': self.best_rewards,
            'solved_prs': self.solved_prs,
            'recent_rewards': list(self.recent_rewards),
            'avg_reward': np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0,
            'attempts_per_pr': self.attempts_per_pr
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingStats':
        """Restore TrainingStats from a serialised dict (round-trip safe)."""
        stats = cls(
            total_steps=data.get('total_steps', 0),
            total_episodes=data.get('total_episodes', 0),
            current_pr_idx=data.get('current_pr_idx', 0),
            current_pr_id=data.get('current_pr_id', ''),
            consecutive_solves=data.get('consecutive_solves', 0),
            best_rewards=data.get('best_rewards', {}),
            solved_prs=data.get('solved_prs', []),
            attempts_per_pr=data.get('attempts_per_pr', {}),
        )
        # Restore recent_rewards as deque
        for r in data.get('recent_rewards', []):
            stats.recent_rewards.append(r)
        return stats


class GRPOTrainer:
    """GRPO trainer using TinyZero/veRL approach.
    
    Key differences from naive GRPO:
    - Token-level log probs and advantages
    - Masked mean for loss computation
    - Group-relative advantage normalization
    - Proper KL penalty handling
    """
    
    def __init__(
        self,
        config: Config,
        policy: LLMPolicy,
        env: CodeEnv,
        pr_tasks: List[PRTask],
        training_logger: Optional[TrainingLogger] = None
    ):
        self.config = config
        self.policy = policy
        self.env = env
        self.pr_tasks = pr_tasks
        
        # Config shortcuts
        self.train_config = config.training
        self.grpo_config = config.training.grpo
        self.curriculum_config = config.curriculum
        
        # GRPO hyperparameters
        self.group_size = self.grpo_config.group_size
        self.beta = self.grpo_config.beta  # KL coefficient
        self.clip_range = self.grpo_config.clip_range
        self.learning_rate = self.train_config.learning_rate
        
        # Setup optimizer — only track trainable (LoRA) parameters to save memory
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy.model.parameters()),
            lr=self.learning_rate
        )
        
        # Cosine LR scheduler with warmup
        # Total budget: max_episodes / group_size gives approximate total steps
        estimated_total_steps = max(self.train_config.max_episodes // self.group_size, 500)
        warmup_steps = min(50, estimated_total_steps // 10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=estimated_total_steps - warmup_steps,
            eta_min=self.learning_rate * 0.1
        )
        self.warmup_steps = warmup_steps
        
        # Stats
        self.stats = TrainingStats()
        
        # Sophisticated experiment logger
        self.exp_logger: Optional[ExperimentLogger] = None
        
        # Legacy logger for backward compat
        self.logger = training_logger or TrainingLogger(name="grpo_trainer")
        
        # Checkpointing
        self.checkpoint_dir = config.checkpoints_path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Callback
        self._ws_callback = None
        
        # Stop flag (set externally to request graceful stop)
        self.stop_requested = False
        
        # Format reward shaping (dense signal for cold-start)
        format_config = FormatRewardConfig(
            enabled=True,
            warmup_steps=50,
            decay_steps=150,
        )
        self.format_scorer = FormatRewardScorer(config=format_config)
        
        # Device
        self.device = next(self.policy.model.parameters()).device
    
    def set_websocket_callback(self, callback):
        """Set callback for real-time updates."""
        self._ws_callback = callback
        
        # Create sophisticated logger with broadcast callback
        self.exp_logger = create_experiment_logger(self.config, callback)
        
        if hasattr(self.logger, 'set_websocket_callback'):
            self.logger.set_websocket_callback(callback)
    
    def _broadcast(self, data: Dict[str, Any]):
        """Broadcast update via callback."""
        if self._ws_callback:
            self._ws_callback(data)
    
    def train(self) -> Dict[str, Any]:
        """Run GRPO training loop."""
        logger.info("Starting GRPO training (TinyZero-style)...")
        logger.info(f"Group size: {self.group_size}, Beta: {self.beta}, Clip: {self.clip_range}")
        logger.info(f"PRs to solve: {[t.pr_id for t in self.pr_tasks]}")

        # Apply data augmentation if enabled
        if self.config.augmentation.enabled:
            try:
                from ..data.augmentation import DataAugmenter
                augmenter = DataAugmenter(seed=self.config.augmentation.seed)
                original_count = len(self.pr_tasks)
                self.pr_tasks = augmenter.augment_all(
                    self.pr_tasks,
                    strategies=self.config.augmentation.strategies,
                    multiplier=self.config.augmentation.multiplier,
                )
                logger.info(
                    f"Augmentation enabled: expanded {original_count} tasks to {len(self.pr_tasks)} tasks"
                )
            except Exception as aug_err:
                logger.warning(f"Augmentation enabled but failed to integrate: {aug_err}")

        self._broadcast({
            'type': 'info',
            'message': f'Starting GRPO training with {len(self.pr_tasks)} PRs'
        })
        
        try:
            while self.stats.current_pr_idx < len(self.pr_tasks):
                # Check if stop has been requested
                if self.stop_requested:
                    logger.info("Stop requested — exiting training loop.")
                    self._broadcast({
                        'type': 'info',
                        'message': 'Training stopped by user'
                    })
                    return {'stats': self.stats.to_dict()}

                current_task = self.pr_tasks[self.stats.current_pr_idx]
                self.stats.current_pr_id = current_task.pr_id

                logger.info(f"Training on PR: {current_task.pr_id}")
                self._broadcast({
                    'type': 'info',
                    'message': f'Training on {current_task.pr_id} ({self.stats.current_pr_idx + 1}/{len(self.pr_tasks)})'
                })

                # When curriculum is disabled, run exactly one training step per PR
                # without solve checking or progression gating
                if not self.curriculum_config.enabled:
                    logger.info(f"Curriculum disabled — running single training step for {current_task.pr_id}")
                    # Global episode budget guard
                    if self.stats.total_episodes >= self.train_config.max_episodes:
                        logger.warning(
                            f"Global episode budget exhausted ({self.train_config.max_episodes}). "
                            f"Stopping training."
                        )
                        return {'stats': self.stats.to_dict()}

                    rollouts = self._collect_rollouts(current_task)
                    if len(rollouts) < 2:
                        logger.warning(
                            f"Only {len(rollouts)}/{self.group_size} rollout(s) collected for "
                            f"{current_task.pr_id}; skipping update (no group signal)"
                        )
                        self.stats.current_pr_idx += 1
                        continue
                    has_signal = self._compute_advantages(rollouts)
                    if not has_signal:
                        logger.info(
                            f"All rollout rewards identical for {current_task.pr_id}; "
                            f"skipping gradient update"
                        )
                        self.stats.current_pr_idx += 1
                        continue
                    try:
                        update_stats = self._grpo_update(rollouts)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower() or 'MPS' in str(e):
                            logger.error(f"OOM during GRPO update at step {self.stats.total_steps}: {e}")
                            self.optimizer.zero_grad()
                            if self.device.type == 'mps':
                                torch.mps.empty_cache()
                            elif self.device.type == 'cuda':
                                torch.cuda.empty_cache()
                            try:
                                self._save_checkpoint()
                            except Exception:
                                pass
                            raise
                        raise
                    self._log_progress(rollouts, update_stats)
                    if self.stats.total_steps % self.train_config.checkpointing.save_every_n_steps == 0:
                        self._save_checkpoint()
                    self.stats.current_pr_idx += 1
                    continue

                # Train until solved (curriculum enabled)
                while not self._is_pr_solved(current_task.pr_id):
                    # Check if stop has been requested
                    if self.stop_requested:
                        logger.info("Stop requested — exiting training loop.")
                        self._broadcast({
                            'type': 'info',
                            'message': 'Training stopped by user'
                        })
                        return {'stats': self.stats.to_dict()}

                    # Global episode budget guard
                    if self.stats.total_episodes >= self.train_config.max_episodes:
                        logger.warning(
                            f"Global episode budget exhausted ({self.train_config.max_episodes}). "
                            f"Stopping training."
                        )
                        self._broadcast({
                            'type': 'info',
                            'message': f'Episode budget exhausted ({self.train_config.max_episodes})'
                        })
                        return {'stats': self.stats.to_dict()}

                    # Per-PR attempt guard (safety valve)
                    pr_attempts = self.stats.attempts_per_pr.get(current_task.pr_id, 0)
                    max_attempts = self.curriculum_config.max_attempts_per_pr
                    if pr_attempts >= max_attempts:
                        logger.warning(
                            f"{current_task.pr_id} hit max attempts ({max_attempts}). "
                            f"Forcing advancement."
                        )
                        self._broadcast({
                            'type': 'info',
                            'message': f'{current_task.pr_id} skipped after {max_attempts} attempts'
                        })
                        break  # Break inner loop, advance to next PR

                    # Collect rollouts (group of completions)
                    rollouts = self._collect_rollouts(current_task)

                    # Need >= 2 rollouts for group-relative advantages; episode
                    # failures can shrink the group. Attempt counters were
                    # already incremented, so the max_attempts valve still
                    # protects against an infinite retry loop.
                    if len(rollouts) < 2:
                        logger.warning(
                            f"Only {len(rollouts)}/{self.group_size} rollout(s) collected for "
                            f"{current_task.pr_id}; retrying (no group signal)"
                        )
                        continue

                    # Compute group-relative advantages
                    has_signal = self._compute_advantages(rollouts)
                    if not has_signal:
                        # All rewards identical (common early on) — gradient
                        # would be exactly zero, so skip the expensive update.
                        logger.info(
                            f"All rollout rewards identical for {current_task.pr_id}; "
                            f"skipping gradient update"
                        )
                        continue

                    # GRPO policy update (with OOM guard)
                    try:
                        update_stats = self._grpo_update(rollouts)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower() or 'MPS' in str(e):
                            logger.error(f"OOM during GRPO update at step {self.stats.total_steps}: {e}")
                            # Clear caches and gradients
                            self.optimizer.zero_grad()
                            if self.device.type == 'mps':
                                torch.mps.empty_cache()
                            elif self.device.type == 'cuda':
                                torch.cuda.empty_cache()
                            # Save emergency checkpoint
                            try:
                                self._save_checkpoint()
                            except Exception:
                                pass
                            raise  # Re-raise so the user sees it
                        raise

                    # Log progress
                    self._log_progress(rollouts, update_stats)

                    # Checkpoint
                    if self.stats.total_steps % self.train_config.checkpointing.save_every_n_steps == 0:
                        self._save_checkpoint()

                    # Periodic mastery evaluation
                    if (self.config.mastery.enabled and
                        self.config.mastery.eval_frequency > 0 and
                        self.stats.total_steps % self.config.mastery.eval_frequency == 0):
                        self._run_mastery_eval()

                # PR solved - advance
                self._broadcast({
                    'type': 'pr_solved',
                    'pr_id': current_task.pr_id,
                    'attempts': self.stats.total_episodes
                })

                # Save milestone checkpoint for solved PR
                self._save_pr_checkpoint(current_task.pr_id)

                self.stats.current_pr_idx += 1
            
            # All PRs solved — run championship round if enabled
            championship_result = None
            if self.config.championship.enabled:
                championship_result = self._run_championship()
            
            result = {'stats': self.stats.to_dict()}
            if championship_result:
                result['championship'] = championship_result
            
            self._broadcast({
                'type': 'training_complete',
                'result': result
            })
            return result
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            self._broadcast({
                'type': 'training_error', 
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise
    
    def _collect_rollouts(self, task: PRTask) -> List[RolloutData]:
        """Collect a group of rollouts for the given task.
        
        Runs full multi-turn episodes where model can:
        1. Read files
        2. Edit files  
        3. Submit for reward
        """
        rollouts = []
        group_id = str(uuid.uuid4())
        max_turns = self.config.environment.max_turns
        
        system_prompt = self._get_system_prompt()
        
        # Generate K episode rollouts
        for i in range(self.group_size):
          try:
            # Reset for new episode
            self.policy.reset_conversation()
            obs = self.env.reset(task.data, task.depends_on)
            
            # Track episode in experiment logger
            episode_num = self.stats.total_episodes + 1
            if self.exp_logger:
                self.exp_logger.start_episode(episode_num, task.pr_id)
            
            # Collect all tokens and log probs across the episode
            turn_data_list: List[TurnData] = []
            all_prompts = []
            
            # Track conversation for multi-turn
            conversation_history = []
            
            # Temperature varies by group member for diversity
            temperature = 0.7 + 0.1 * i
            
            # Run episode until terminal or max turns
            turn = 0
            terminal = False
            
            while not terminal and turn < max_turns:
                turn += 1
                
                # Build prompt with history
                current_prompt = self._build_turn_prompt(
                    task, obs, conversation_history, system_prompt
                )
                all_prompts.append(current_prompt)
                
                # Generate response.
                # Note: we do NOT use generation-time log probs (output.log_probs):
                # those come from post-processed sampling scores (temperature,
                # top-k/top-p filtering, renormalisation), which is a different
                # distribution from the raw model log-softmax used in the GRPO
                # update. old_log_probs are recomputed below with the exact same
                # method and context as the update's new_log_probs, so the
                # importance ratio is exactly 1.0 before the first optimizer step.
                output = self.policy.generate(
                    current_prompt,
                    system_prompt=system_prompt if turn == 1 else None,
                    return_log_probs=False,
                    temperature=temperature
                )
                
                # Broadcast generation
                self._broadcast({
                    'type': 'generation_token',
                    'pr_id': task.pr_id,
                    'turn': turn,
                    'group_idx': i + 1,
                    'full_text': output.text
                })
                
                # Collect tokens and log probs for this turn
                response_ids = torch.tensor(output.token_ids, device=self.device)

                # Use the EXACT input ids that generation saw (chat template
                # applied + truncation), not a re-tokenisation of the raw
                # prompt — otherwise old/new log probs are computed over
                # different contexts and the importance ratio is garbage.
                turn_prompt_ids = torch.tensor(
                    output.prompt_token_ids, device=self.device
                )

                # Recompute old log probs under the raw model distribution,
                # matching _compute_token_log_probs_with_grad in the update.
                lp = self._compute_token_log_probs(turn_prompt_ids, response_ids)
                
                # Store per-turn data (prompt + response + log probs)
                turn_data_list.append(TurnData(
                    prompt_ids=turn_prompt_ids.to(self.device),
                    response_ids=response_ids,
                    old_log_probs=lp.detach()
                ))
                
                # Execute action in environment
                action = self.env.parse_action(output.text)
                
                # Broadcast tool call
                if action.tool_name:
                    self._broadcast({
                        'type': 'tool_call',
                        'tool': action.tool_name,
                        'args': action.tool_args,
                        'pr_id': task.pr_id,
                        'turn': turn
                    })
                
                obs = self.env.step(action)
                terminal = obs.is_terminal
                
                # Broadcast tool result
                if action.tool_name:
                    tool_success = obs.info.get('success', True) if obs.info else True
                    self._broadcast({
                        'type': 'tool_result',
                        'tool': action.tool_name,
                        'success': tool_success,
                        'output': obs.content[:200] if obs.content else '',
                        'terminal': terminal
                    })
                    # Log tool call in experiment logger
                    if self.exp_logger:
                        self.exp_logger.log_tool_call(
                            tool=action.tool_name,
                            args=action.tool_args or {},
                            result=obs.content[:500] if obs.content else '',
                            success=tool_success
                        )
                
                # Log generation in experiment logger
                if self.exp_logger:
                    self.exp_logger.log_generation(output.text, turn)
                
                # Add to conversation history
                conversation_history.append({
                    'role': 'assistant',
                    'content': output.text
                })
                if not terminal:
                    conversation_history.append({
                        'role': 'user', 
                        'content': obs.content
                    })
            
            # Get episode result
            episode = self.env.get_episode()
            task_reward = episode.total_reward if episode else 0.0
            solved = episode.solved if episode else False
            
            # Clean up working directory from this episode to prevent disk leak
            self.env.cleanup()
            
            # Compute format shaping reward (dense signal for cold-start)
            assistant_outputs = [h['content'] for h in conversation_history if h['role'] == 'assistant']
            format_reward = self.format_scorer.score_episode(
                turn_outputs=assistant_outputs,
                training_step=self.stats.total_steps,
            )
            
            # Combined reward: task reward + format shaping (clamped to [0, 1])
            reward = min(1.0, task_reward + format_reward)
            
            # Log episode end in experiment logger
            if self.exp_logger:
                self.exp_logger.end_episode(
                    reward=reward,
                    solved=solved,
                    num_turns=turn,
                    error=None
                )
                # Also log the breakdown as extra metrics
                self.exp_logger.log_metrics({
                    'task_reward': task_reward,
                    'format_reward': format_reward,
                    'combined_reward': reward,
                    'num_turns': float(turn),
                }, step=self.stats.total_steps, episode=self.stats.total_episodes + 1)
            
            # Compute response mask (all tokens valid)
            total_response_tokens = sum(td.response_ids.shape[0] for td in turn_data_list)
            if total_response_tokens > 0:
                response_mask = torch.ones(total_response_tokens, device=self.device)
            else:
                response_mask = torch.zeros(1, device=self.device)
            
            # Broadcast episode completion
            full_conversation = "\n---\n".join([h['content'] for h in conversation_history if h['role'] == 'assistant'])
            self._broadcast({
                'type': 'generation_complete',
                'pr_id': task.pr_id,
                'episode': self.stats.total_episodes + 1,
                'turn': self.stats.total_episodes + 1,  # For UI compatibility
                'group_idx': i + 1,
                'turns': turn,
                'reward': reward,
                'solved': solved,
                'full_text': full_conversation
            })
            
            rollouts.append(RolloutData(
                turns=turn_data_list,
                response_mask=response_mask,
                reward=reward,
                solved=solved,
                group_id=group_id,
                response_text="\n---\n".join([h['content'] for h in conversation_history if h['role'] == 'assistant']),
                task_reward=task_reward
            ))
            
            # Update stats
            self.stats.total_episodes += 1
            self.stats.recent_rewards.append(reward)
            
            # Track attempts per PR
            if task.pr_id not in self.stats.attempts_per_pr:
                self.stats.attempts_per_pr[task.pr_id] = 0
            self.stats.attempts_per_pr[task.pr_id] += 1
            
            # Track best TASK reward (excluding format shaping) so the solve
            # threshold can't be crossed by warm-up shaping alone
            if task_reward > self.stats.best_rewards.get(task.pr_id, 0):
                self.stats.best_rewards[task.pr_id] = task_reward
            
            # Track consecutive solves
            if solved:
                self.stats.consecutive_solves += 1
            else:
                self.stats.consecutive_solves = 0
            
            # Broadcast episode result
            self._broadcast({
                'type': 'episode',
                'episode': self.stats.total_episodes,
                'pr_id': task.pr_id,
                'reward': reward,
                'solved': solved,
                'turns': turn,
                'message': f'Episode {self.stats.total_episodes}: {task.pr_id} - R={reward:.2f} ({turn} turns)'
            })
          except Exception as ep_err:
            # Per-episode error handling: log and skip, don't kill entire run
            logger.warning(f"Episode {i+1}/{self.group_size} failed for {task.pr_id}: {ep_err}")
            self._broadcast({
                'type': 'warning',
                'message': f'Episode {i+1} failed: {str(ep_err)[:200]}'
            })
            # Log failed episode in experiment logger
            if self.exp_logger:
                self.exp_logger.end_episode(
                    reward=0.0,
                    solved=False,
                    num_turns=0,
                    error=str(ep_err)[:500]
                )
            continue
        
        # Free rollout-phase memory before GRPO update
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        
        return rollouts
    
    def _build_turn_prompt(
        self, 
        task: PRTask, 
        obs: Any, 
        history: List[Dict[str, str]],
        system_prompt: str
    ) -> str:
        """Build prompt for a turn in multi-turn episode.
        
        Includes conversation history so the model has context of prior
        tool calls and their results.
        """
        if not history:
            # First turn - return observation content (task description)
            return obs.content
        else:
            # Subsequent turns - include recent history for context
            # Keep last N turns to avoid context overflow
            max_history_turns = 6  # 3 assistant + 3 user messages
            recent_history = history[-max_history_turns:]
            
            parts = []
            for msg in recent_history:
                role_label = "Assistant" if msg['role'] == 'assistant' else "Result"
                content = msg['content']
                # Truncate very long results to save context
                if msg['role'] == 'user' and len(content) > 500:
                    content = content[:500] + "\n... (truncated)"
                parts.append(f"[{role_label}]\n{content}")
            
            parts.append(f"[Result]\n{obs.content}")
            parts.append("\nContinue with the next action:")
            
            return "\n\n".join(parts)
    
    def _gather_token_log_probs(
        self, 
        log_probs: torch.Tensor, 
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Gather log probs for specific token IDs.
        
        Args:
            log_probs: (seq_len, vocab_size) or (seq_len,) if already gathered
            token_ids: (seq_len,) token IDs to gather
            
        Returns:
            (seq_len,) tensor of log probs for each token
        """
        if len(log_probs.shape) == 1:
            return log_probs
        
        # Ensure tensors are on same device
        log_probs = log_probs.to(self.device)
        token_ids = token_ids.to(self.device)
        
        # Vectorized gather: for each position, get log prob of the actual token
        seq_len = min(log_probs.shape[0], token_ids.shape[0])
        log_probs = log_probs[:seq_len]
        ids = token_ids[:seq_len].clamp(0, log_probs.shape[1] - 1)
        gathered = log_probs[torch.arange(seq_len, device=self.device), ids]
        
        return gathered
    
    def _compute_token_log_probs(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log probs for a response given prompt.
        
        This is used when we don't have log probs from generation.
        """
        # Ensure both tensors are on the same device
        prompt_ids = prompt_ids.to(self.device)
        response_ids = response_ids.to(self.device)
        
        # Concatenate prompt and response
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.policy.model(full_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)
        
        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probs for response tokens
        prompt_len = prompt_ids.shape[0]
        response_log_probs = []
        
        for i in range(len(response_ids)):
            # Log prob at position (prompt_len + i - 1) predicts token at (prompt_len + i)
            if prompt_len + i - 1 >= 0 and prompt_len + i < full_ids.shape[1]:
                token_id = response_ids[i]
                lp = log_probs[prompt_len + i - 1, token_id]
                response_log_probs.append(lp)
        
        if response_log_probs:
            return torch.stack(response_log_probs)
        return torch.zeros(1, device=self.device)
    
    def _compute_advantages(self, rollouts: List[RolloutData]) -> bool:
        """Compute group-relative advantages (in-place).

        Following TinyZero: normalize rewards within each group,
        then spread the advantage to all tokens in the response.

        Returns:
            True if at least one group produced a non-zero advantage
            (i.e. there is a learning signal worth doing an update for).
        """
        # Group rollouts by group_id
        groups = defaultdict(list)
        for rollout in rollouts:
            groups[rollout.group_id].append(rollout)

        has_signal = False

        # For each group, compute normalized advantages
        for group_id, group_rollouts in groups.items():
            rewards = torch.tensor([r.reward for r in group_rollouts], device=self.device)

            # A group needs >= 2 rollouts and reward variance to carry signal.
            # (torch.std on a single element is NaN; equal rewards give 0/eps.)
            if len(group_rollouts) < 2:
                logger.warning(
                    f"Group {group_id[:8]} has only {len(group_rollouts)} rollout(s); "
                    f"zeroing advantages (no relative signal)"
                )
                for rollout in group_rollouts:
                    rollout.advantage = 0.0
                continue

            # Normalize within group (population std: n in denominator)
            mean_reward = rewards.mean()
            std_reward = rewards.std(unbiased=False)

            if std_reward.item() < 1e-6:
                # All rewards identical — no relative signal in this group
                for rollout in group_rollouts:
                    rollout.advantage = 0.0
                continue

            normalized = (rewards - mean_reward) / (std_reward + 1e-8)
            has_signal = True

            # Assign advantage to each rollout
            # Store as attribute (will be used in update)
            for i, rollout in enumerate(group_rollouts):
                # Spread scalar advantage to all response tokens
                rollout.advantage = normalized[i].item()

        return has_signal
    
    def _grpo_update(self, rollouts: List[RolloutData]) -> Dict[str, float]:
        """Perform GRPO policy update using token-level computation.
        
        Correctly handles multi-turn episodes by recomputing log probs
        per-turn with the matching prompt context.
        
        Uses per-turn micro-backward to keep only 1 forward pass's
        computation graph in memory at a time (prevents MPS OOM).
        """
        self.optimizer.zero_grad()
        
        # Count total items for averaging the accumulated gradients
        num_rollouts = len(rollouts)
        
        # Stats accumulators (no gradients needed for these)
        total_pg_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        total_tokens = 0
        clip_fracs = []
        all_ratios = []
        all_advantages = []
        accumulated_loss = 0.0  # scalar for logging
        
        for rollout in rollouts:
            advantage = rollout.advantage
            rollout_pg_stat = 0.0
            rollout_kl_stat = 0.0
            rollout_tokens = 0
            rollout_entropy = 0.0
            rollout_clip_count = 0.0
            rollout_total_clip_denom = 0.0
            turn_ratios = []
            
            for turn in rollout.turns:
                # Recompute log probs for THIS turn with its correct prompt context
                new_log_probs = self._compute_token_log_probs_with_grad(
                    turn.prompt_ids,
                    turn.response_ids
                )
                
                # Align lengths (should match, but be safe)
                min_len = min(new_log_probs.shape[0], turn.old_log_probs.shape[0])
                if min_len == 0:
                    continue
                
                new_lp = new_log_probs[:min_len]
                old_lp = turn.old_log_probs[:min_len].to(self.device)
                mask = torch.ones(min_len, device=self.device)
                n_tokens = mask.sum()
                
                # Compute ratio per token
                ratio = torch.exp(new_lp - old_lp)
                
                # Clipped surrogate objective (per token)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
                pg_loss_tokens = -torch.min(surr1, surr2)
                
                # Mean over tokens in this turn
                turn_pg = (pg_loss_tokens * mask).sum() / (n_tokens + 1e-8)
                
                # KL penalty per turn — clamped to prevent explosion
                log_ratio = old_lp - new_lp
                log_ratio_clamped = torch.clamp(log_ratio, -5.0, 5.0)
                kl_tokens = torch.exp(log_ratio_clamped) - 1.0 - log_ratio_clamped
                turn_kl = (kl_tokens * mask).sum() / (n_tokens + 1e-8)
                
                # This turn's loss
                turn_loss = turn_pg + self.beta * turn_kl
                
                # Scale by token weight relative to total, divided by num_rollouts
                # so the accumulated gradient ~ mean over rollouts of mean over tokens
                # We'll do a simple: backward(loss / num_rollouts) per turn
                # and gradient = sum of these = mean-of-rollouts loss
                scaled_loss = turn_loss / num_rollouts
                
                # ---- MICRO BACKWARD: free this turn's graph immediately ----
                scaled_loss.backward()
                # After backward(), the computation graph from this forward pass
                # is freed, keeping peak memory = 1 forward pass at a time.
                
                # Accumulate scalar stats (detached, no graph)
                with torch.no_grad():
                    rollout_pg_stat += turn_pg.item() * n_tokens.item()
                    rollout_kl_stat += turn_kl.item() * n_tokens.item()
                    rollout_tokens += n_tokens.item()
                    accumulated_loss += turn_loss.item()
                    
                    # Entropy proxy
                    token_entropy = -(new_lp * mask).sum() / (n_tokens + 1e-8)
                    rollout_entropy += token_entropy.item() * n_tokens.item()
                    
                    # Clip fraction
                    clipped = (surr2 > surr1).float()
                    rollout_clip_count += (clipped * mask).sum().item()
                    rollout_total_clip_denom += n_tokens.item()
                    
                    # Ratio for diagnostics
                    turn_ratios.append((ratio * mask).sum().item() / (n_tokens.item() + 1e-8))
                
                # Explicitly free MPS memory after each turn
                del new_log_probs, new_lp, old_lp, mask, ratio, surr1, surr2
                del pg_loss_tokens, turn_pg, kl_tokens, turn_kl, turn_loss, scaled_loss
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
            
            # Rollout-level stats
            if rollout_tokens > 0:
                total_pg_loss += rollout_pg_stat / rollout_tokens
                total_kl += rollout_kl_stat / rollout_tokens
                total_tokens += rollout_tokens
                total_entropy += rollout_entropy / rollout_tokens
                
                clip_fracs.append(rollout_clip_count / (rollout_total_clip_denom + 1e-8))
                all_ratios.append(np.mean(turn_ratios) if turn_ratios else 1.0)
                all_advantages.append(advantage)
        
        # Gradient clipping (only trainable parameters)
        trainable_params = [p for p in self.policy.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        
        # NaN guard: skip optimizer step if gradients exploded
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        if not np.isfinite(grad_norm_val):
            logger.warning(f"Step {self.stats.total_steps}: NaN/Inf grad_norm ({grad_norm_val}), skipping optimizer step")
            self.optimizer.zero_grad()
            self.stats.total_steps += 1
            return {
                'loss': 0.0, 'pg_loss': 0.0, 'kl_loss': total_kl / max(num_rollouts, 1),
                'entropy': 0.0, 'clip_frac': 0.0, 'grad_norm': 0.0,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'mean_reward': np.mean([r.reward for r in rollouts]),
                'max_reward': max(r.reward for r in rollouts),
                'min_reward': min(r.reward for r in rollouts),
                'solve_rate': sum(1 for r in rollouts if r.solved) / len(rollouts),
                'mean_advantage': 0.0, 'mean_ratio': 1.0,
                'num_tokens': int(total_tokens), 'num_rollouts': num_rollouts,
                'skipped': True
            }
        
        # Optimizer step
        self.optimizer.step()
        
        # LR scheduler step (with linear warmup)
        if self.stats.total_steps < self.warmup_steps:
            warmup_factor = (self.stats.total_steps + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.learning_rate * warmup_factor
        else:
            self.scheduler.step()
        
        self.stats.total_steps += 1
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'loss': accumulated_loss / max(num_rollouts, 1),
            'pg_loss': total_pg_loss / max(num_rollouts, 1),
            'kl_loss': total_kl / max(num_rollouts, 1),
            'entropy': total_entropy / max(num_rollouts, 1),
            'clip_frac': np.mean(clip_fracs) if clip_fracs else 0.0,
            'grad_norm': grad_norm_val,
            'learning_rate': current_lr,
            'mean_reward': np.mean([r.reward for r in rollouts]),
            'max_reward': max(r.reward for r in rollouts),
            'min_reward': min(r.reward for r in rollouts),
            'solve_rate': sum(1 for r in rollouts if r.solved) / len(rollouts),
            'mean_advantage': np.mean(all_advantages) if all_advantages else 0.0,
            'mean_ratio': np.mean(all_ratios) if all_ratios else 1.0,
            'num_tokens': int(total_tokens),
            'num_rollouts': num_rollouts
        }
    
    def _compute_token_log_probs_with_grad(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log probs WITH gradients for policy update.
        
        Memory-optimised: only computes log_softmax over the response
        positions rather than the full sequence.
        """
        prompt_ids = prompt_ids.to(self.device)
        response_ids = response_ids.to(self.device)
        
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        prompt_len = prompt_ids.shape[0]
        resp_len = response_ids.shape[0]
        
        if resp_len == 0:
            return torch.zeros(1, device=self.device, requires_grad=True)
        
        # Forward pass (WITH gradients)
        outputs = self.policy.model(full_ids)
        
        # Slice logits to only the positions that predict response tokens
        # Position (prompt_len - 1) predicts response_ids[0],
        # Position (prompt_len + resp_len - 2) predicts response_ids[-1]
        start_pos = max(prompt_len - 1, 0)
        end_pos = prompt_len + resp_len - 1  # exclusive end
        resp_logits = outputs.logits[0, start_pos:end_pos, :]  # (resp_len, vocab)
        
        # Drop reference to full logits to free memory
        del outputs
        
        # log_softmax only over response positions → much smaller allocation
        resp_log_probs = F.log_softmax(resp_logits, dim=-1)  # (resp_len, vocab)
        del resp_logits
        
        # Gather the log probs for the actual response token ids
        token_log_probs = resp_log_probs[
            torch.arange(resp_len, device=self.device),
            response_ids[:resp_len]
        ]
        del resp_log_probs
        
        return token_log_probs
    
    def _is_pr_solved(self, pr_id: str) -> bool:
        """Check if PR is considered solved."""
        if pr_id in self.stats.solved_prs:
            return True
        
        best_reward = self.stats.best_rewards.get(pr_id, 0)
        threshold = self.curriculum_config.solve_threshold
        min_solves = self.curriculum_config.min_consecutive_solves
        
        if best_reward >= threshold and self.stats.consecutive_solves >= min_solves:
            self.stats.solved_prs.append(pr_id)
            logger.info(f"PR {pr_id} solved! Best reward: {best_reward:.3f}")
            return True
        
        return False
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent.

        Phase-aware: returns different prompts based on the environment type.
        """
        # Check if we're in a phase env that doesn't need tool instructions
        from ..environment.phase_env import PhaseOneEnv, PhaseTwoEnv, PhaseThreeEnv
        if isinstance(self.env, PhaseOneEnv):
            return "You are a Python developer. Output ONLY valid Python code. No explanations, no markdown fences, no tool calls."
        if isinstance(self.env, PhaseTwoEnv):
            return """You are a code assistant. Write updated files using this EXACT format:

<file path="filename.py">
...complete file content...
</file>

Output ONLY the <file> block. No explanations, no markdown."""

        if isinstance(self.env, PhaseThreeEnv):
            return """You are a code assistant. You complete tasks in 2 steps:

Step 1 — Read the file:
<tool>read_file(path="filename.py")</tool>

Step 2 — Write the complete updated file:
<file path="filename.py">
...complete updated file content...
</file>

Rules:
1. First read the file to see its current content
2. Then write the COMPLETE updated file wrapped in <file> tags
3. Include ALL existing code plus your additions
4. Do NOT call submit() or write_file()"""

        return """You are a code assistant. You solve tasks by reading files, writing changes, and submitting.

Tools available:
- <tool>read_file(path="...")</tool> — read a file
- <tool>submit()</tool> — submit your solution when done

To write a complete file, use this format:
<file path="filename.py">
...complete file content...
</file>

Example workflow:
1. Read the file:
<tool>read_file(path="pyutils/strings.py")</tool>

2. Write the updated file with your changes:
<file path="pyutils/strings.py">
...complete updated file content here...
</file>

3. Submit when done:
<tool>submit()</tool>

Rules:
1. Read files before modifying them
2. Write the COMPLETE file content (all existing code plus your additions)
3. Call submit() when finished"""
    
    def _log_progress(self, rollouts: List[RolloutData], update_stats: Dict[str, float]):
        """Log training progress using sophisticated logger."""
        # Always emit a console-visible step summary (CLI runs have no
        # exp_logger/websocket, so this is their only step-level signal)
        logger.info(
            f"Step {self.stats.total_steps}: loss={update_stats['loss']:.4f} "
            f"pg={update_stats['pg_loss']:.4f} kl={update_stats['kl_loss']:.4f} "
            f"ratio={update_stats.get('mean_ratio', 1.0):.4f} "
            f"grad_norm={update_stats.get('grad_norm', 0):.3f} "
            f"reward={update_stats['mean_reward']:.3f} "
            f"solve_rate={update_stats['solve_rate']:.2f}"
        )

        # Use experiment logger if available
        if self.exp_logger:
            self.exp_logger.log_step(
                step=self.stats.total_steps,
                loss=update_stats['loss'],
                pg_loss=update_stats['pg_loss'],
                kl_loss=update_stats['kl_loss'],
                entropy=update_stats.get('entropy', 0),
                grad_norm=update_stats.get('grad_norm', 0),
                learning_rate=update_stats.get('learning_rate', 0),
                clip_frac=update_stats['clip_frac'],
                mean_reward=update_stats['mean_reward'],
                max_reward=update_stats['max_reward'],
                solve_rate=update_stats['solve_rate'],
                num_episodes=len(rollouts),
                extra={
                    'min_reward': update_stats.get('min_reward', 0),
                    'mean_advantage': update_stats.get('mean_advantage', 0),
                    'mean_ratio': update_stats.get('mean_ratio', 1.0),
                    'num_tokens': update_stats.get('num_tokens', 0),
                    'num_rollouts': update_stats.get('num_rollouts', 0),
                    'pr_id': self.stats.current_pr_id,
                    'avg_reward_100': np.mean(list(self.stats.recent_rewards)) if self.stats.recent_rewards else 0
                }
            )
        else:
            # Fallback to simple broadcast
            self._broadcast({
                'type': 'step',
                'step': self.stats.total_steps,
                'metrics': {
                    'loss': update_stats['loss'],
                    'pg_loss': update_stats['pg_loss'],
                    'kl_loss': update_stats['kl_loss'],
                    'entropy': update_stats.get('entropy', 0),
                    'clip_frac': update_stats['clip_frac'],
                    'learning_rate': update_stats.get('learning_rate', 0),
                    'mean_reward': update_stats['mean_reward'],
                    'max_reward': update_stats['max_reward'],
                    'min_reward': update_stats.get('min_reward', 0),
                    'solve_rate': update_stats['solve_rate'],
                    'mean_advantage': update_stats.get('mean_advantage', 0),
                    'mean_ratio': update_stats.get('mean_ratio', 1.0),
                    'avg_reward': np.mean(list(self.stats.recent_rewards)) if self.stats.recent_rewards else 0
                }
            })
            
            if self.stats.total_steps % 10 == 0:
                logger.info(
                    f"Step {self.stats.total_steps} | "
                    f"PR: {self.stats.current_pr_id} | "
                    f"Loss: {update_stats['loss']:.4f} | "
                    f"PG: {update_stats['pg_loss']:.4f} | "
                    f"KL: {update_stats['kl_loss']:.4f} | "
                    f"LR: {update_stats.get('learning_rate', 0):.2e} | "
                    f"Reward: {update_stats['mean_reward']:.3f} | "
                    f"Solve: {update_stats['solve_rate']:.1%}"
                )
    
    def _save_checkpoint(self):
        """Save training checkpoint and clean up old ones."""
        checkpoint_path = self.checkpoint_dir / f"step_{self.stats.total_steps}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.policy.save(str(checkpoint_path / "model"))
        
        # Save stats
        with open(checkpoint_path / "stats.json", 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        # Save optimizer state
        if self.train_config.checkpointing.save_optimizer:
            torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        # Save scheduler state
        if self.train_config.checkpointing.save_scheduler:
            torch.save(self.scheduler.state_dict(), checkpoint_path / "scheduler.pt")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        self._broadcast({
            'type': 'checkpoint',
            'path': str(checkpoint_path),
            'step': self.stats.total_steps
        })
        
        # Clean up old checkpoints (respect keep_last_n)
        keep_last_n = self.train_config.checkpointing.keep_last_n
        if keep_last_n > 0:
            self._cleanup_old_checkpoints(keep_last_n)
    
    def _save_pr_checkpoint(self, pr_id: str):
        """Save a milestone checkpoint when a PR is solved. These are never auto-cleaned."""
        # Normalise PR id for directory name (e.g. "PR-001" -> "pr_001")
        safe_name = pr_id.lower().replace("-", "_")
        checkpoint_path = self.checkpoint_dir / f"solved_{safe_name}_step_{self.stats.total_steps}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.policy.save(str(checkpoint_path / "model"))
        
        # Save stats + PR metadata
        with open(checkpoint_path / "stats.json", 'w') as f:
            json.dump({
                **self.stats.to_dict(),
                'solved_pr': pr_id,
                'milestone': True,
            }, f, indent=2)
        
        logger.info(f"🏆 PR milestone checkpoint saved: {checkpoint_path}")
        self._broadcast({
            'type': 'checkpoint',
            'path': str(checkpoint_path),
            'step': self.stats.total_steps,
            'milestone': True,
            'pr_id': pr_id,
        })
    
    def _cleanup_old_checkpoints(self, keep_last_n: int):
        """Remove old checkpoints, keeping only the most recent N.
        
        Milestone checkpoints (solved_*) are never deleted.
        """
        # Find only regular step checkpoints (step_*), skip milestones (solved_*)
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0
        )
        
        # Remove oldest checkpoints if we exceed the limit
        while len(checkpoints) > keep_last_n:
            oldest = checkpoints.pop(0)
            try:
                shutil.rmtree(oldest)
                logger.info(f"Removed old checkpoint: {oldest}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {oldest}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a full checkpoint: model weights, optimizer, scheduler, and training stats.
        
        Args:
            checkpoint_path: Path to checkpoint directory or model subdirectory.
                             Accepts either 'checkpoints/step_100' or 'checkpoints/step_100/model'.
        """
        ckpt = Path(checkpoint_path)
        
        # If user pointed at the model subdir, go up one level
        if ckpt.name == "model" and (ckpt.parent / "stats.json").exists():
            ckpt = ckpt.parent
        
        # Load model weights (LoRA adapter)
        model_path = ckpt / "model"
        if model_path.exists():
            self.policy.load_checkpoint(str(model_path))
            logger.info(f"Restored model from {model_path}")
        else:
            # Maybe the path itself IS the model dir
            self.policy.load_checkpoint(str(ckpt))
            logger.info(f"Restored model from {ckpt}")

        # Loading the adapter creates new parameter tensors, so the optimizer
        # and scheduler built in __init__ now reference dead parameters.
        # Rebuild them BEFORE restoring their state dicts.
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy.model.parameters()),
            lr=self.learning_rate
        )
        estimated_total_steps = max(self.train_config.max_episodes // self.group_size, 500)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=estimated_total_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.1
        )

        # Restore training stats
        stats_path = ckpt / "stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)
            self.stats = TrainingStats.from_dict(stats_data)
            logger.info(
                f"Restored stats: step={self.stats.total_steps}, "
                f"episodes={self.stats.total_episodes}, "
                f"pr_idx={self.stats.current_pr_idx}, "
                f"solved={self.stats.solved_prs}"
            )
        
        # Restore optimizer state
        optimizer_path = ckpt / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logger.info("Restored optimizer state")
        
        # Restore scheduler state
        scheduler_path = ckpt / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
            logger.info("Restored scheduler state")
    
    def _run_championship(self) -> Dict[str, Any]:
        """Run championship round: model must solve ALL PRs from base repo.

        Respects config:
        - ``mode``: "sequential" means PRs must be solved in order (stop on
          first failure unless retries are allowed). Any other value runs all
          PRs independently.
        - ``allow_retries``: when False each PR gets exactly 1 attempt. When
          True the PR is retried up to ``max_attempts_per_pr`` times.

        Returns:
            Dict with championship results.
        """
        logger.info("\n" + "=" * 60)
        logger.info("CHAMPIONSHIP ROUND")
        logger.info("=" * 60)
        self._broadcast({'type': 'info', 'message': 'Starting championship round...'})

        results = []
        all_passed = True
        system_prompt = self._get_system_prompt()
        max_turns = self.config.environment.max_turns
        threshold = self.curriculum_config.solve_threshold
        champ_config = self.config.championship
        sequential = champ_config.mode == "sequential"
        allow_retries = champ_config.allow_retries
        max_retry_attempts = self.curriculum_config.max_attempts_per_pr if allow_retries else 1

        for task in self.pr_tasks:
            passed = False
            best_reward = 0.0
            best_turns = 0

            for attempt in range(max_retry_attempts):
                self.policy.reset_conversation()
                obs = self.env.reset(task.data, task.depends_on)

                conversation_history = []
                turn = 0
                terminal = False

                while not terminal and turn < max_turns:
                    turn += 1
                    prompt = self._build_turn_prompt(task, obs, conversation_history, system_prompt)
                    output = self.policy.generate(
                        prompt,
                        system_prompt=system_prompt if turn == 1 else None,
                        return_log_probs=False,
                        temperature=0.0  # Greedy for championship
                    )

                    action = self.env.parse_action(output.text)
                    obs = self.env.step(action)
                    terminal = obs.is_terminal

                    conversation_history.append({'role': 'assistant', 'content': output.text})
                    if not terminal:
                        conversation_history.append({'role': 'user', 'content': obs.content})

                episode = self.env.get_episode()
                reward = episode.total_reward if episode else 0.0

                self.env.cleanup()

                if reward > best_reward:
                    best_reward = reward
                    best_turns = turn

                if reward >= threshold:
                    passed = True
                    best_reward = reward
                    best_turns = turn
                    break  # No need to retry

            if not passed:
                all_passed = False

            results.append({
                'pr_id': task.pr_id,
                'reward': best_reward,
                'passed': passed,
                'turns': best_turns,
            })

            status = 'PASS' if passed else 'FAIL'
            logger.info(f"  {status} {task.pr_id}: reward={best_reward:.3f} ({best_turns} turns)")
            self._broadcast({
                'type': 'championship_pr',
                'pr_id': task.pr_id, 'reward': best_reward,
                'passed': passed, 'turns': best_turns,
            })

            # In sequential mode, stop on first failure (no point continuing)
            if sequential and not passed:
                logger.info(f"Sequential championship: stopping after {task.pr_id} failed")
                # Mark remaining PRs as not attempted
                remaining_idx = self.pr_tasks.index(task) + 1
                for remaining_task in self.pr_tasks[remaining_idx:]:
                    results.append({
                        'pr_id': remaining_task.pr_id,
                        'reward': 0.0,
                        'passed': False,
                        'turns': 0,
                    })
                break
        
        championship_result = {
            'passed': all_passed,
            'results': results,
            'prs_passed': sum(1 for r in results if r['passed']),
            'prs_total': len(results),
        }
        
        # Save championship transcript if configured
        if self.config.championship.save_transcript:
            transcript_path = self.checkpoint_dir / "championship_transcript.json"
            with open(transcript_path, 'w') as f:
                json.dump(championship_result, f, indent=2)
        
        status = '🏆 PASSED' if all_passed else '❌ FAILED'
        logger.info(f"\nChampionship {status}: {championship_result['prs_passed']}/{championship_result['prs_total']} PRs")
        self._broadcast({
            'type': 'championship_complete',
            'result': championship_result
        })
        
        # Log via training logger if available
        if hasattr(self.logger, 'log_championship'):
            self.logger.log_championship(championship_result)
        
        return championship_result
    
    def _run_mastery_eval(self) -> Dict[str, Any]:
        """Run mastery evaluation: test solved PRs to confirm the model still solves them.
        
        This catches catastrophic forgetting — the model might solve PR-003 but
        forget how to solve PR-001 after further training.
        
        Returns:
            Dict with per-PR mastery results.
        """
        mastery_config = self.config.mastery
        test_config = mastery_config.test_config
        system_prompt = self._get_system_prompt()
        max_turns = self.config.environment.max_turns
        threshold = test_config.success_threshold_per_pr
        
        logger.info(f"Running mastery evaluation at step {self.stats.total_steps}...")
        self._broadcast({'type': 'info', 'message': f'Mastery eval at step {self.stats.total_steps}'})
        
        # Determine which PRs to evaluate
        if test_config.require_all_prs:
            # All solved PRs must pass mastery
            prs_to_eval = [t for t in self.pr_tasks if t.pr_id in self.stats.solved_prs]
        else:
            # Only check the most recently solved PR (subset mode)
            prs_to_eval = []
            for t in reversed(self.pr_tasks):
                if t.pr_id in self.stats.solved_prs:
                    prs_to_eval = [t]
                    break

        results = {}
        for task in prs_to_eval:
            successes = 0
            for attempt in range(test_config.num_attempts_per_pr):
                self.policy.reset_conversation()
                obs = self.env.reset(task.data, task.depends_on)
                conversation_history = []
                turn = 0
                terminal = False

                while not terminal and turn < max_turns:
                    turn += 1
                    prompt = self._build_turn_prompt(task, obs, conversation_history, system_prompt)
                    output = self.policy.generate(
                        prompt,
                        system_prompt=system_prompt if turn == 1 else None,
                        return_log_probs=False,
                        temperature=0.2  # Low temperature for eval
                    )
                    action = self.env.parse_action(output.text)
                    obs = self.env.step(action)
                    terminal = obs.is_terminal
                    conversation_history.append({'role': 'assistant', 'content': output.text})
                    if not terminal:
                        conversation_history.append({'role': 'user', 'content': obs.content})

                episode = self.env.get_episode()
                reward = episode.total_reward if episode else 0.0
                self.env.cleanup()

                if reward >= threshold:
                    successes += 1

            results[task.pr_id] = {
                'successes': successes,
                'attempts': test_config.num_attempts_per_pr,
                'pass_rate': successes / test_config.num_attempts_per_pr,
                'mastered': successes == test_config.num_attempts_per_pr,
            }

            status = 'PASS' if results[task.pr_id]['mastered'] else 'WARN'
            logger.info(f"  {status} {task.pr_id}: {successes}/{test_config.num_attempts_per_pr}")

        # Check overall mastery against the configured threshold
        if results:
            mastered_count = sum(1 for r in results.values() if r['mastered'])
            total_evaluated = len(results)
            mastery_fraction = mastered_count / total_evaluated
            all_mastered = mastery_fraction >= test_config.overall_success_threshold
        else:
            all_mastered = False
        
        self._broadcast({
            'type': 'mastery_eval',
            'step': self.stats.total_steps,
            'results': results,
            'all_mastered': all_mastered,
        })
        
        # Save model on mastery if configured
        if all_mastered and self.config.model_saving.save_on_mastery:
            mastery_path = self.checkpoint_dir / f"mastery_step_{self.stats.total_steps}"
            mastery_path.mkdir(parents=True, exist_ok=True)
            self.policy.save(str(mastery_path / "model"))
            logger.info(f"🎓 Mastery checkpoint saved: {mastery_path}")
        
        return results
