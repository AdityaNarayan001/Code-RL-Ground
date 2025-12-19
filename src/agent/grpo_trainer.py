"""GRPO (Group Relative Policy Optimization) trainer for code generation RL.

GRPO is a simpler alternative to PPO that:
1. Generates multiple completions per prompt (group)
2. Uses relative ranking within the group for advantage estimation
3. Doesn't require a separate value function/critic
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..utils.config import Config
from ..utils.logging import TrainingLogger, get_logger
from ..environment import CodeEnv
from ..data import PRTask
from .policy import LLMPolicy

logger = get_logger(__name__)


@dataclass
class CompletionData:
    """Data for a single completion."""
    prompt: str
    response: str
    reward: float
    log_prob: float
    solved: bool


@dataclass 
class GroupData:
    """Data for a group of completions from same prompt."""
    prompt: str
    completions: List[CompletionData]
    
    @property
    def rewards(self) -> List[float]:
        return [c.reward for c in self.completions]
    
    @property
    def log_probs(self) -> List[float]:
        return [c.log_prob for c in self.completions]
    
    @property
    def advantages(self) -> List[float]:
        """Compute relative advantages within group."""
        rewards = np.array(self.rewards)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        return ((rewards - mean_reward) / std_reward).tolist()


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
    """GRPO trainer for code generation RL.
    
    Implements Group Relative Policy Optimization:
    - Generate K completions per prompt
    - Rank by reward within group
    - Use relative advantage for policy update
    - No value function needed
    """
    
    def __init__(
        self,
        config: Config,
        policy: LLMPolicy,
        env: CodeEnv,
        pr_tasks: List[PRTask],
        training_logger: Optional[TrainingLogger] = None
    ):
        """Initialize GRPO trainer.
        
        Args:
            config: Configuration
            policy: LLM policy
            env: Code environment
            pr_tasks: PR tasks in curriculum order
            training_logger: Training logger
        """
        self.config = config
        self.policy = policy
        self.env = env
        self.pr_tasks = pr_tasks
        
        # Config shortcuts
        self.train_config = config.training
        self.grpo_config = config.training.grpo
        self.curriculum_config = config.curriculum
        
        # GRPO hyperparameters (access dataclass attributes directly)
        self.group_size = self.grpo_config.group_size
        self.beta = self.grpo_config.beta  # KL penalty
        self.clip_range = self.grpo_config.clip_range
        self.learning_rate = self.train_config.learning_rate
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.model.parameters(),
            lr=self.learning_rate
        )
        
        # Stats
        self.stats = TrainingStats()
        
        # Logger
        self.logger = training_logger or TrainingLogger(config)
        
        # Checkpointing
        self.checkpoint_dir = config.checkpoints_path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Callback
        self._ws_callback = None
    
    def set_websocket_callback(self, callback):
        """Set callback for real-time updates."""
        self._ws_callback = callback
        if hasattr(self.logger, 'set_websocket_callback'):
            self.logger.set_websocket_callback(callback)
    
    def _broadcast(self, data: Dict[str, Any]):
        """Broadcast update via callback."""
        if self._ws_callback:
            self._ws_callback(data)
    
    def train(self) -> Dict[str, Any]:
        """Run GRPO training loop.
        
        Returns:
            Training results
        """
        logger.info("Starting GRPO training...")
        logger.info(f"Group size: {self.group_size}")
        logger.info(f"Beta (KL penalty): {self.beta}")
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
                    # Collect group of completions
                    group = self._collect_group(current_task)
                    
                    # GRPO update
                    update_stats = self._grpo_update(group)
                    
                    # Log
                    self._log_progress(group, update_stats)
                    
                    # Checkpoint
                    if self.stats.total_steps % self.train_config.checkpointing.save_every_n_steps == 0:
                        self._save_checkpoint()
                    
                    # Safety valve
                    if self.stats.total_episodes > self.curriculum_config.max_attempts_per_pr * (self.stats.current_pr_idx + 1):
                        logger.warning(f"Max attempts for {current_task.pr_id}, moving on")
                        break
                
                # Mark solved and advance
                if current_task.pr_id not in self.stats.solved_prs:
                    self.stats.solved_prs.append(current_task.pr_id)
                    self._broadcast({
                        'type': 'pr_solved',
                        'pr_id': current_task.pr_id
                    })
                
                self.stats.current_pr_idx += 1
                self.stats.consecutive_solves = 0
            
            # Championship round
            if self.config.championship.enabled:
                self._run_championship()
            
            # Save final
            if self.config.model_saving.save_final:
                self._save_final_model()
            
            return {
                'success': True,
                'total_episodes': self.stats.total_episodes,
                'solved_prs': self.stats.solved_prs,
                'stats': self.stats.to_dict()
            }
            
        except KeyboardInterrupt:
            logger.info("Training interrupted")
            self._save_checkpoint()
            return {
                'success': False,
                'reason': 'interrupted',
                'stats': self.stats.to_dict()
            }
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self._broadcast({
                'type': 'training_error',
                'error': str(e)
            })
            return {
                'success': False,
                'reason': str(e),
                'stats': self.stats.to_dict()
            }
    
    def _collect_group(self, task: PRTask) -> GroupData:
        """Collect a group of completions for the same prompt.
        
        Args:
            task: Current PR task
            
        Returns:
            Group data with multiple completions
        """
        completions = []
        
        # Reset env once to get the prompt
        obs = self.env.reset(task.data, task.depends_on)
        prompt = obs.content
        system_prompt = self._get_system_prompt()
        
        # Generate K completions
        for i in range(self.group_size):
            # Reset policy conversation
            self.policy.reset_conversation()
            
            # Generate with different sampling
            output = self.policy.generate(
                prompt,
                system_prompt=system_prompt,
                return_log_probs=True,
                temperature=0.7 + 0.1 * i  # Vary temperature for diversity
            )
            
            # Broadcast generation for UI display
            self._broadcast({
                'type': 'generation_complete',
                'pr_id': task.pr_id,
                'turn': i + 1,
                'full_text': output.text
            })
            
            # Execute in environment
            self.env.reset(task.data, task.depends_on)
            action = self.env.parse_action(output.text)
            obs = self.env.step(action)
            
            # Get reward
            episode = self.env.get_episode()
            reward = episode.total_reward
            solved = episode.solved
            
            # Compute total log prob from log_probs tensor
            total_log_prob = 0.0
            if output.log_probs is not None:
                total_log_prob = output.log_probs.sum().item()
            
            completions.append(CompletionData(
                prompt=prompt,
                response=output.text,
                reward=reward,
                log_prob=total_log_prob,
                solved=solved
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
            
            # Log episode
            self._broadcast({
                'type': 'episode',
                'episode': self.stats.total_episodes,
                'pr_id': task.pr_id,
                'reward': reward,
                'solved': solved
            })
            
            self.env.cleanup()
        
        return GroupData(prompt=prompt, completions=completions)
    
    def _grpo_update(self, group: GroupData) -> Dict[str, float]:
        """Perform GRPO policy update.
        
        GRPO Loss = -E[advantage * log_prob] + beta * KL
        
        Where advantage is computed relative to group mean.
        
        Args:
            group: Group of completions
            
        Returns:
            Update statistics
        """
        self.optimizer.zero_grad()
        
        advantages = group.advantages
        old_log_probs = group.log_probs
        
        total_loss = 0.0
        policy_losses = []
        
        for i, completion in enumerate(group.completions):
            # Recompute log prob under current policy (returns tensor of per-token log probs)
            new_log_prob_tensor = self.policy.compute_log_prob(
                group.prompt,
                completion.response
            )
            # Sum to get total log probability (scalar)
            new_log_prob = new_log_prob_tensor.sum()
            
            # old_log_probs[i] is already a scalar (total log prob)
            old_log_prob = torch.tensor(old_log_probs[i], device=new_log_prob.device)
            
            # Compute ratio (in log space, then exp)
            ratio = torch.exp(new_log_prob - old_log_prob)
            
            # Clipped objective (like PPO)
            advantage = advantages[i]
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
            policy_loss = -torch.min(surr1, surr2)
            
            # KL penalty (simplified - difference in log probs)
            kl_penalty = (old_log_prob - new_log_prob).abs()
            
            loss = policy_loss + self.beta * kl_penalty
            total_loss += loss
            policy_losses.append(policy_loss.item())
        
        # Average loss over group
        total_loss = total_loss / self.group_size
        
        # Backward and step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.stats.total_steps += 1
        
        return {
            'loss': total_loss.item(),
            'mean_advantage': np.mean(advantages),
            'mean_reward': np.mean(group.rewards),
            'max_reward': np.max(group.rewards),
            'solve_rate': sum(1 for c in group.completions if c.solved) / len(group.completions)
        }
    
    def _log_progress(self, group: GroupData, update_stats: Dict[str, float]):
        """Log training progress."""
        self._broadcast({
            'type': 'step',
            'step': self.stats.total_steps,
            'metrics': {
                'loss': update_stats['loss'],
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
                f"Mean R: {update_stats['mean_reward']:.3f} | "
                f"Solve: {update_stats['solve_rate']:.1%}"
            )
    
    def _is_pr_solved(self, pr_id: str) -> bool:
        """Check if PR meets solve criteria."""
        if not self.curriculum_config.strict_progression:
            return True
        
        best = self.stats.best_rewards.get(pr_id, 0)
        consecutive = self.stats.consecutive_solves
        
        return (
            best >= self.curriculum_config.solve_threshold and
            consecutive >= self.curriculum_config.min_consecutive_solves
        )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are a code agent that solves programming tasks.
You have access to tools to read, write, and execute code.

Available tools:
- read_file(path): Read contents of a file
- write_file(path, content): Write content to a file  
- edit_file(path, old_text, new_text): Replace text in a file
- run_python(code): Execute Python code
- search_code(pattern): Search for pattern in codebase
- submit(): Submit your solution

Respond with a tool call in this format:
<tool>tool_name</tool>
<args>{"arg": "value"}</args>

After making changes, use submit() to check if your solution is correct."""
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.stats.total_steps}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.policy.model.save_pretrained(checkpoint_path / "lora")
        
        # Save training state
        state = {
            'stats': self.stats.to_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path / "trainer_state.pt")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        self._broadcast({
            'type': 'checkpoint',
            'step': self.stats.total_steps,
            'path': str(checkpoint_path)
        })
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only last N checkpoints."""
        keep_n = self.train_config.checkpointing.keep_last_n
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*"))
        
        for old_ckpt in checkpoints[:-keep_n]:
            import shutil
            shutil.rmtree(old_ckpt)
    
    def _save_final_model(self):
        """Save final model."""
        final_path = Path(self.config.model_saving.final_model_path)
        final_path.mkdir(parents=True, exist_ok=True)
        
        self.policy.model.save_pretrained(final_path)
        self.policy.tokenizer.save_pretrained(final_path)
        
        logger.info(f"Final model saved: {final_path}")
    
    def _run_championship(self):
        """Run championship round."""
        logger.info("Starting championship round...")
        self._broadcast({
            'type': 'info',
            'message': 'Starting championship round!'
        })
        
        passed = 0
        for task in self.pr_tasks:
            # Single attempt per PR
            obs = self.env.reset(task.data, task.depends_on)
            self.policy.reset_conversation()
            
            output = self.policy.generate(
                obs.content,
                system_prompt=self._get_system_prompt()
            )
            
            action = self.env.parse_action(output.text)
            self.env.step(action)
            episode = self.env.get_episode()
            
            if episode.solved:
                passed += 1
                logger.info(f"Championship: {task.pr_id} PASSED")
            else:
                logger.info(f"Championship: {task.pr_id} FAILED")
            
            self.env.cleanup()
        
        result = passed == len(self.pr_tasks)
        logger.info(f"Championship: {passed}/{len(self.pr_tasks)} - {'PASSED' if result else 'FAILED'}")
        
        self._broadcast({
            'type': 'championship_complete',
            'passed': passed,
            'total': len(self.pr_tasks),
            'success': result
        })
        
        return result
    
    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint_path = Path(path)
        
        # Load LoRA
        from peft import PeftModel
        self.policy.model = PeftModel.from_pretrained(
            self.policy.model,
            checkpoint_path / "lora"
        )
        
        # Load training state
        state = torch.load(checkpoint_path / "trainer_state.pt")
        self.optimizer.load_state_dict(state['optimizer'])
        
        # Restore stats
        stats_dict = state['stats']
        self.stats.total_steps = stats_dict['total_steps']
        self.stats.total_episodes = stats_dict['total_episodes']
        self.stats.current_pr_idx = stats_dict['current_pr_idx']
        self.stats.solved_prs = stats_dict['solved_prs']
        self.stats.best_rewards = stats_dict['best_rewards']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
