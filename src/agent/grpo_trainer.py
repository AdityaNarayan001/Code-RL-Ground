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

from ..utils.config import Config
from ..utils.logging import TrainingLogger, get_logger
from ..utils.metrics import ExperimentLogger, create_experiment_logger
from ..environment import CodeEnv
from ..data import PRTask
from .policy import LLMPolicy

logger = get_logger(__name__)


@dataclass
class RolloutData:
    """Data from a single rollout."""
    prompt_ids: torch.Tensor          # (prompt_len,)
    response_ids: torch.Tensor        # (response_len,)
    response_mask: torch.Tensor       # (response_len,) - 1 for valid tokens
    old_log_probs: torch.Tensor       # (response_len,) - per-token log probs
    reward: float                     # scalar outcome reward
    solved: bool
    group_id: str                     # to group responses from same prompt
    response_text: str                # for display


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'current_pr_idx': self.current_pr_idx,
            'current_pr_id': self.current_pr_id,
            'consecutive_solves': self.consecutive_solves,
            'best_rewards': self.best_rewards,
            'solved_prs': self.solved_prs,
            'avg_reward': np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0
        }


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
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.model.parameters(),
            lr=self.learning_rate
        )
        
        # Stats
        self.stats = TrainingStats()
        
        # Sophisticated experiment logger
        self.exp_logger: Optional[ExperimentLogger] = None
        
        # Legacy logger for backward compat
        self.logger = training_logger or TrainingLogger(config)
        
        # Checkpointing
        self.checkpoint_dir = config.checkpoints_path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Callback
        self._ws_callback = None
        
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
        
        self._broadcast({
            'type': 'info',
            'message': f'Starting GRPO training with {len(self.pr_tasks)} PRs'
        })
        
        try:
            while self.stats.current_pr_idx < len(self.pr_tasks):
                current_task = self.pr_tasks[self.stats.current_pr_idx]
                self.stats.current_pr_id = current_task.pr_id
                
                logger.info(f"Training on PR: {current_task.pr_id}")
                self._broadcast({
                    'type': 'info',
                    'message': f'Training on {current_task.pr_id} ({self.stats.current_pr_idx + 1}/{len(self.pr_tasks)})'
                })
                
                # Train until solved
                while not self._is_pr_solved(current_task.pr_id):
                    # Collect rollouts (group of completions)
                    rollouts = self._collect_rollouts(current_task)
                    
                    # Compute group-relative advantages
                    self._compute_advantages(rollouts)
                    
                    # GRPO policy update
                    update_stats = self._grpo_update(rollouts)
                    
                    # Log progress
                    self._log_progress(rollouts, update_stats)
                    
                    # Checkpoint
                    if self.stats.total_steps % self.train_config.checkpointing.save_every_n_steps == 0:
                        self._save_checkpoint()
                
                # PR solved - advance
                self._broadcast({
                    'type': 'pr_solved',
                    'pr_id': current_task.pr_id,
                    'attempts': self.stats.total_episodes
                })
                self.stats.current_pr_idx += 1
            
            # All PRs solved
            self._broadcast({
                'type': 'training_complete',
                'result': {'stats': self.stats.to_dict()}
            })
            return {'stats': self.stats.to_dict()}
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
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
            # Reset for new episode
            self.policy.reset_conversation()
            obs = self.env.reset(task.data, task.depends_on)
            
            # Collect all tokens and log probs across the episode
            all_response_ids = []
            all_log_probs = []
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
                
                # Generate response
                output = self.policy.generate(
                    current_prompt,
                    system_prompt=system_prompt if turn == 1 else None,
                    return_log_probs=True,
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
                
                # Collect tokens and log probs
                response_ids = torch.tensor(output.token_ids, device=self.device)
                all_response_ids.append(response_ids)
                
                # Get log probs for this turn's tokens
                if output.log_probs is not None:
                    prompt_ids = self.policy.tokenizer(
                        current_prompt, return_tensors="pt", truncation=True
                    )['input_ids'][0]
                    if len(output.log_probs.shape) > 1:
                        lp = self._gather_token_log_probs(output.log_probs, response_ids)
                    else:
                        lp = output.log_probs
                    all_log_probs.append(lp)
                else:
                    prompt_ids = self.policy.tokenizer(
                        current_prompt, return_tensors="pt", truncation=True
                    )['input_ids'][0]
                    lp = self._compute_token_log_probs(prompt_ids, response_ids)
                    all_log_probs.append(lp)
                
                # Execute action in environment
                action = self.env.parse_action(output.text)
                obs = self.env.step(action)
                terminal = obs.is_terminal
                
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
            reward = episode.total_reward if episode else 0.0
            solved = episode.solved if episode else False
            
            # Concatenate all response tokens
            if all_response_ids:
                combined_response_ids = torch.cat(all_response_ids)
                combined_log_probs = torch.cat(all_log_probs)
                response_mask = torch.ones(len(combined_response_ids), device=self.device)
            else:
                combined_response_ids = torch.zeros(1, dtype=torch.long, device=self.device)
                combined_log_probs = torch.zeros(1, device=self.device)
                response_mask = torch.zeros(1, device=self.device)
            
            # Use first prompt's token ids as base (for advantage computation)
            first_prompt = all_prompts[0] if all_prompts else ""
            prompt_ids = self.policy.tokenizer(
                first_prompt, return_tensors="pt", truncation=True
            )['input_ids'][0]
            
            # Broadcast episode completion
            self._broadcast({
                'type': 'generation_complete',
                'pr_id': task.pr_id,
                'episode': self.stats.total_episodes + 1,
                'group_idx': i + 1,
                'turns': turn,
                'reward': reward,
                'solved': solved
            })
            
            rollouts.append(RolloutData(
                prompt_ids=prompt_ids.to(self.device),
                response_ids=combined_response_ids,
                response_mask=response_mask,
                old_log_probs=combined_log_probs.detach(),
                reward=reward,
                solved=solved,
                group_id=group_id,
                response_text="\n---\n".join([h['content'] for h in conversation_history if h['role'] == 'assistant'])
            ))
            
            # Update stats
            self.stats.total_episodes += 1
            self.stats.recent_rewards.append(reward)
            
            if reward > self.stats.best_rewards.get(task.pr_id, 0):
                self.stats.best_rewards[task.pr_id] = reward
            
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
        
        return rollouts
    
    def _build_turn_prompt(
        self, 
        task: PRTask, 
        obs: Any, 
        history: List[Dict[str, str]],
        system_prompt: str
    ) -> str:
        """Build prompt for a turn in multi-turn episode."""
        if not history:
            # First turn - return observation content
            return obs.content
        else:
            # Subsequent turns - return observation (tool result/feedback)
            return obs.content
    
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
        
        # Gather: for each position, get log prob of the actual token
        seq_len = min(log_probs.shape[0], token_ids.shape[0])
        gathered = torch.zeros(seq_len, device=self.device)
        
        for i in range(seq_len):
            if i < log_probs.shape[0] and token_ids[i] < log_probs.shape[1]:
                gathered[i] = log_probs[i, token_ids[i]]
        
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
    
    def _compute_advantages(self, rollouts: List[RolloutData]) -> None:
        """Compute group-relative advantages (in-place).
        
        Following TinyZero: normalize rewards within each group,
        then spread the advantage to all tokens in the response.
        """
        # Group rollouts by group_id
        groups = defaultdict(list)
        for rollout in rollouts:
            groups[rollout.group_id].append(rollout)
        
        # For each group, compute normalized advantages
        for group_id, group_rollouts in groups.items():
            rewards = torch.tensor([r.reward for r in group_rollouts], device=self.device)
            
            # Normalize within group
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            normalized = (rewards - mean_reward) / std_reward
            
            # Assign advantage to each rollout
            # Store as attribute (will be used in update)
            for i, rollout in enumerate(group_rollouts):
                # Spread scalar advantage to all response tokens
                rollout.advantage = normalized[i].item()
    
    def _grpo_update(self, rollouts: List[RolloutData]) -> Dict[str, float]:
        """Perform GRPO policy update using token-level computation."""
        self.optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_pg_loss = 0.0
        total_kl = 0.0
        total_tokens = 0
        clip_fracs = []
        
        for rollout in rollouts:
            # Recompute log probs under current policy (with gradients)
            new_log_probs = self._compute_token_log_probs_with_grad(
                rollout.prompt_ids,
                rollout.response_ids
            )
            
            # Align lengths
            min_len = min(new_log_probs.shape[0], rollout.old_log_probs.shape[0])
            if min_len == 0:
                continue
                
            new_lp = new_log_probs[:min_len]
            old_lp = rollout.old_log_probs[:min_len].to(self.device)
            mask = rollout.response_mask[:min_len].to(self.device)
            
            # Compute ratio per token
            ratio = torch.exp(new_lp - old_lp)
            
            # Advantage (same for all tokens in this response)
            advantage = rollout.advantage
            
            # Clipped surrogate objective (per token)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
            pg_loss_tokens = -torch.min(surr1, surr2)
            
            # Masked mean over tokens
            pg_loss = (pg_loss_tokens * mask).sum() / (mask.sum() + 1e-8)
            
            # KL penalty (low-variance version from TinyZero)
            # KL = old_lp - new_lp (approximation)
            kl_tokens = old_lp - new_lp
            kl_loss = (kl_tokens.abs() * mask).sum() / (mask.sum() + 1e-8)
            
            # Combined loss
            loss = pg_loss + self.beta * kl_loss
            total_loss = total_loss + loss
            
            # Stats
            total_pg_loss += pg_loss.detach().item()
            total_kl += kl_loss.detach().item()
            total_tokens += mask.sum().item()
            
            # Clip fraction
            clipped = (surr2 > surr1).float()
            clip_fracs.append((clipped * mask).sum().item() / (mask.sum().item() + 1e-8))
        
        # Average over rollouts
        num_rollouts = len(rollouts)
        if num_rollouts > 0:
            total_loss = total_loss / num_rollouts
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(), 1.0
        )
        
        # Optimizer step
        self.optimizer.step()
        
        self.stats.total_steps += 1
        
        return {
            'loss': total_loss.detach().item(),
            'pg_loss': total_pg_loss / max(num_rollouts, 1),
            'kl_loss': total_kl / max(num_rollouts, 1),
            'clip_frac': np.mean(clip_fracs) if clip_fracs else 0.0,
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'mean_reward': np.mean([r.reward for r in rollouts]),
            'max_reward': max(r.reward for r in rollouts),
            'solve_rate': sum(1 for r in rollouts if r.solved) / len(rollouts)
        }
    
    def _compute_token_log_probs_with_grad(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log probs WITH gradients for policy update."""
        # Ensure both tensors are on the same device
        prompt_ids = prompt_ids.to(self.device)
        response_ids = response_ids.to(self.device)
        
        # Concatenate prompt and response
        full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
        
        # Forward pass (WITH gradients)
        outputs = self.policy.model(full_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        
        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probs for response tokens
        prompt_len = prompt_ids.shape[0]
        response_log_probs = []
        
        for i in range(len(response_ids)):
            # Log prob at position (prompt_len + i - 1) predicts token at (prompt_len + i)
            pos = prompt_len + i - 1
            if pos >= 0 and pos < logits.shape[0]:
                token_id = response_ids[i]
                lp = log_probs[pos, token_id]
                response_log_probs.append(lp)
        
        if response_log_probs:
            return torch.stack(response_log_probs)
        return torch.zeros(1, device=self.device, requires_grad=True)
    
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
        """Get system prompt for the agent with few-shot examples."""
        return """You are a skilled software engineer. Your task is to implement the requested code changes.

IMPORTANT: You MUST use the exact tool format shown below. Do NOT use markdown code blocks.

## Tool Format
<tool>tool_name(param="value")</tool>

## Available Tools
- read_file(path) - Read a file
- write_file(path, content) - Write a new file  
- edit_file(path, old_content, new_content) - Edit existing file
- run_python(code) - Run Python code
- submit() - Submit your solution

## Example: Adding a function to a file

1. First, read the file:
<tool>read_file(path="utils/math.py")</tool>

2. Then edit it to add new code:
<tool>edit_file(path="utils/math.py", old_content="def add(a, b):\\n    return a + b", new_content="def add(a, b):\\n    return a + b\\n\\ndef multiply(a, b):\\n    return a * b")</tool>

3. Submit when done:
<tool>submit()</tool>

Now complete the task using ONLY this exact <tool>...</tool> format. No explanations needed - just tool calls."""
    
    def _log_progress(self, rollouts: List[RolloutData], update_stats: Dict[str, float]):
        """Log training progress using sophisticated logger."""
        # Use experiment logger if available
        if self.exp_logger:
            self.exp_logger.log_step(
                step=self.stats.total_steps,
                loss=update_stats['loss'],
                pg_loss=update_stats['pg_loss'],
                kl_loss=update_stats['kl_loss'],
                grad_norm=update_stats.get('grad_norm', 0),
                clip_frac=update_stats['clip_frac'],
                mean_reward=update_stats['mean_reward'],
                max_reward=update_stats['max_reward'],
                solve_rate=update_stats['solve_rate'],
                num_episodes=len(rollouts)
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
                    'clip_frac': update_stats['clip_frac'],
                    'mean_reward': update_stats['mean_reward'],
                    'max_reward': update_stats['max_reward'],
                    'solve_rate': update_stats['solve_rate'],
                    'avg_reward': np.mean(list(self.stats.recent_rewards)) if self.stats.recent_rewards else 0
                }
            })
            
            if self.stats.total_steps % 10 == 0:
                logger.info(
                    f"Step {self.stats.total_steps} | "
                    f"PR: {self.stats.current_pr_id} | "
                    f"Loss: {update_stats['loss']:.4f} | "
                    f"Reward: {update_stats['mean_reward']:.3f} | "
                    f"Solve: {update_stats['solve_rate']:.1%}"
                )
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"step_{self.stats.total_steps}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.policy.save(str(checkpoint_path / "model"))
        
        # Save stats
        import json
        with open(checkpoint_path / "stats.json", 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        self._broadcast({
            'type': 'checkpoint',
            'path': str(checkpoint_path),
            'step': self.stats.total_steps
        })
