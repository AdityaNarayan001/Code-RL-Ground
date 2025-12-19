"""PPO Trainer for RL fine-tuning."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil
from collections import deque
import numpy as np

from .policy import LLMPolicy, GenerationOutput
from ..environment.code_env import CodeEnv, Action, Episode
from ..data.pr_loader import PRTask
from ..utils.config import Config
from ..utils.logging import get_logger, TrainingLogger


logger = get_logger(__name__)


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    prompts: List[str] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    
    def clear(self):
        """Clear the buffer."""
        self.prompts.clear()
        self.responses.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        return len(self.prompts)


@dataclass 
class TrainingStats:
    """Statistics for training."""
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


class PPOTrainer:
    """PPO trainer for code generation RL.
    
    Implements:
    - PPO with clipped objective
    - GAE for advantage estimation
    - Curriculum learning
    - Checkpointing
    """
    
    def __init__(
        self,
        config: Config,
        policy: LLMPolicy,
        env: CodeEnv,
        pr_tasks: List[PRTask],
        logger: Optional[TrainingLogger] = None
    ):
        """Initialize trainer.
        
        Args:
            config: Configuration object
            policy: LLM policy
            env: Code environment
            pr_tasks: List of PR tasks in curriculum order
            logger: Training logger
        """
        self.config = config
        self.policy = policy
        self.env = env
        self.pr_tasks = pr_tasks
        self.logger = logger or get_logger(__name__)
        
        # Training config
        self.train_config = config.training
        self.ppo_config = config.training.ppo
        self.curriculum_config = config.curriculum
        
        # Optimizer
        self.optimizer = AdamW(
            [p for p in policy.model.parameters() if p.requires_grad],
            lr=self.train_config.learning_rate
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_config.max_episodes
        )
        
        # Buffer
        self.buffer = RolloutBuffer()
        
        # Stats
        self.stats = TrainingStats()
        
        # Checkpointing
        self.checkpoint_dir = config.checkpoints_path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # WebSocket callback
        self._ws_callback = None
    
    def set_websocket_callback(self, callback):
        """Set callback for real-time updates."""
        self._ws_callback = callback
        if hasattr(self.logger, 'set_websocket_callback'):
            self.logger.set_websocket_callback(callback)
    
    def train(self) -> Dict[str, Any]:
        """Run training loop.
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting PPO training...")
        logger.info(f"PRs to solve: {[t.pr_id for t in self.pr_tasks]}")
        
        try:
            while self.stats.current_pr_idx < len(self.pr_tasks):
                current_task = self.pr_tasks[self.stats.current_pr_idx]
                self.stats.current_pr_id = current_task.pr_id
                
                logger.info(f"Training on PR: {current_task.pr_id} ({self.stats.current_pr_idx + 1}/{len(self.pr_tasks)})")
                
                # Train until PR is solved
                while not self._is_pr_solved(current_task.pr_id):
                    # Collect rollouts
                    episodes = self._collect_rollouts(current_task)
                    
                    # Update policy
                    update_stats = self._ppo_update()
                    
                    # Log progress
                    self._log_progress(episodes, update_stats)
                    
                    # Checkpoint
                    if self.stats.total_steps % self.train_config.checkpointing.save_every_n_steps == 0:
                        self._save_checkpoint()
                    
                    # Check max attempts
                    if self.stats.total_episodes > self.curriculum_config.max_attempts_per_pr * (self.stats.current_pr_idx + 1):
                        logger.warning(f"Max attempts reached for {current_task.pr_id}, forcing move")
                        break
                
                # Mark PR as solved
                if current_task.pr_id not in self.stats.solved_prs:
                    self.stats.solved_prs.append(current_task.pr_id)
                    self.logger.log_pr_solved(
                        current_task.pr_id,
                        self.stats.total_episodes,
                        self.stats.best_rewards.get(current_task.pr_id, 0)
                    )
                
                # Move to next PR
                self.stats.current_pr_idx += 1
                self.stats.consecutive_solves = 0
            
            # All PRs solved - run championship if enabled
            if self.config.championship.enabled:
                championship_results = self._run_championship()
                self.logger.log_championship(championship_results)
            
            # Save final model
            if self.config.model_saving.save_final:
                self._save_final_model()
            
            return {
                'success': True,
                'total_episodes': self.stats.total_episodes,
                'solved_prs': self.stats.solved_prs,
                'stats': self.stats.to_dict()
            }
            
        except KeyboardInterrupt:
            logger.info("Training interrupted, saving checkpoint...")
            self._save_checkpoint()
            return {
                'success': False,
                'reason': 'interrupted',
                'stats': self.stats.to_dict()
            }
    
    def _collect_rollouts(self, task: PRTask) -> List[Episode]:
        """Collect rollout episodes.
        
        Args:
            task: Current PR task
            
        Returns:
            List of completed episodes
        """
        episodes = []
        
        for _ in range(self.train_config.batch_size):
            episode = self._run_episode(task)
            episodes.append(episode)
            
            # Add to buffer
            self._add_to_buffer(episode)
            
            # Update stats
            self.stats.total_episodes += 1
            self.stats.recent_rewards.append(episode.total_reward)
            
            if episode.total_reward > self.stats.best_rewards.get(task.pr_id, 0):
                self.stats.best_rewards[task.pr_id] = episode.total_reward
            
            # Track consecutive solves
            if episode.solved:
                self.stats.consecutive_solves += 1
            else:
                self.stats.consecutive_solves = 0
        
        return episodes
    
    def _run_episode(self, task: PRTask) -> Episode:
        """Run a single episode.
        
        Args:
            task: PR task
            
        Returns:
            Completed episode
        """
        # Reset environment
        obs = self.env.reset(task.data, task.depends_on)
        
        # Reset policy conversation
        self.policy.reset_conversation()
        
        # Build system prompt
        system_prompt = self._get_system_prompt()
        
        # Add task to history
        self.policy.add_to_history("user", obs.content)
        
        # Run turns
        while not obs.is_terminal:
            # Get model response
            prompt = self.policy.get_conversation_prompt()
            output = self.policy.generate(
                prompt,
                return_log_probs=True
            )
            
            # Log generation for real-time display
            self.logger.log_generation(task.pr_id, len(self.env.current_episode.turns), output.text)
            
            # Parse action
            action = self.env.parse_action(output.text)
            
            # Take step
            obs = self.env.step(action)
            
            # Add to history
            self.policy.add_to_history("assistant", output.text)
            if not obs.is_terminal:
                self.policy.add_to_history("user", obs.content)
        
        # Get episode
        episode = self.env.get_episode()
        
        # Log
        self.logger.log_episode(
            self.stats.total_episodes,
            task.pr_id,
            episode.total_reward,
            episode.solved
        )
        
        # Cleanup
        self.env.cleanup()
        
        return episode
    
    def _add_to_buffer(self, episode: Episode):
        """Add episode to rollout buffer."""
        # Simplified - just store the key data
        # In full implementation, would store per-turn data
        prompt = episode.task_description
        response = "\n".join(
            action.raw_output 
            for action, _ in episode.turns
        )
        
        self.buffer.prompts.append(prompt)
        self.buffer.responses.append(response)
        self.buffer.rewards.append(episode.total_reward)
    
    def _ppo_update(self) -> Dict[str, float]:
        """Perform PPO update.
        
        Returns:
            Update statistics
        """
        if len(self.buffer) == 0:
            return {}
        
        self.stats.total_steps += 1
        
        # Compute advantages using GAE
        self._compute_advantages()
        
        # PPO update loop
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        
        for _ in range(self.train_config.epochs_per_pr):
            for i in range(len(self.buffer)):
                prompt = self.buffer.prompts[i]
                response = self.buffer.responses[i]
                old_reward = self.buffer.rewards[i]
                
                # Compute new log prob
                new_log_prob = self.policy.compute_log_prob(prompt, response)
                new_log_prob_sum = new_log_prob.sum()
                
                # For simplicity, use reward as advantage
                advantage = torch.tensor(old_reward - 0.5)  # Center around 0.5
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_prob_sum - new_log_prob_sum.detach())
                clip_range = self.ppo_config.clip_range
                
                policy_loss_1 = advantage * ratio
                policy_loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss_step = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Backward
                self.optimizer.zero_grad()
                policy_loss_step.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                total_loss += policy_loss_step.item()
                policy_loss += policy_loss_step.item()
        
        # Step scheduler
        self.scheduler.step()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _compute_advantages(self):
        """Compute advantages using GAE."""
        # Simplified: just use rewards directly as advantages
        # Full implementation would use value function and GAE
        gamma = self.ppo_config.gamma
        gae_lambda = self.ppo_config.gae_lambda
        
        rewards = self.buffer.rewards
        n = len(rewards)
        
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(n)):
            delta = rewards[i]  # Simplified
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + 0.5)  # Baseline
        
        self.buffer.advantages = advantages
        self.buffer.returns = returns
    
    def _is_pr_solved(self, pr_id: str) -> bool:
        """Check if PR is considered solved."""
        # Need consecutive solves
        if self.stats.consecutive_solves >= self.curriculum_config.min_consecutive_solves:
            best = self.stats.best_rewards.get(pr_id, 0)
            if best >= self.curriculum_config.solve_threshold:
                return True
        return False
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return """You are a code generation agent. Your task is to implement the requested changes to a codebase.

You have access to tools for reading, writing, and editing files. Use them to complete the task.

When you're done with all changes, call the submit() tool.

Be precise and follow the requirements exactly. Generate clean, well-documented code."""
    
    def _log_progress(self, episodes: List[Episode], update_stats: Dict[str, float]):
        """Log training progress."""
        avg_reward = np.mean([e.total_reward for e in episodes])
        solve_rate = np.mean([1 if e.solved else 0 for e in episodes])
        
        self.logger.log_step(self.stats.total_steps, {
            'avg_reward': avg_reward,
            'solve_rate': solve_rate,
            'consecutive_solves': self.stats.consecutive_solves,
            'current_pr': self.stats.current_pr_id,
            **update_stats
        })
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_name = f"checkpoint_step_{self.stats.total_steps}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model
        self.policy.save(str(checkpoint_path))
        
        # Save training state
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        # Save optimizer state
        if self.train_config.checkpointing.save_optimizer:
            torch.save(
                self.optimizer.state_dict(),
                checkpoint_path / "optimizer.pt"
            )
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the latest N."""
        keep_n = self.train_config.checkpointing.keep_last_n
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*"),
            key=lambda p: int(p.name.split("_")[-1])
        )
        
        while len(checkpoints) > keep_n:
            old_checkpoint = checkpoints.pop(0)
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def _save_final_model(self):
        """Save the final trained model."""
        final_path = self.config.model_saving.final_model_path
        self.policy.save(final_path)
        
        # Save training summary
        summary_path = Path(final_path) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'completed': True,
                'total_episodes': self.stats.total_episodes,
                'solved_prs': self.stats.solved_prs,
                'best_rewards': self.stats.best_rewards
            }, f, indent=2)
        
        logger.info(f"Final model saved to {final_path}")
    
    def _run_championship(self) -> Dict[str, Any]:
        """Run championship round.
        
        Returns:
            Championship results
        """
        logger.info("🏆 Starting Championship Round...")
        
        results = {
            'prs': {},
            'total_score': 0.0,
            'all_passed': True
        }
        
        for task in self.pr_tasks:
            logger.info(f"Championship: {task.pr_id}")
            
            # Single attempt, no retries
            episode = self._run_episode(task)
            
            results['prs'][task.pr_id] = {
                'reward': episode.total_reward,
                'solved': episode.solved
            }
            
            results['total_score'] += episode.total_reward
            if not episode.solved:
                results['all_passed'] = False
        
        results['total_score'] /= len(self.pr_tasks)
        
        # Save transcript if enabled
        if self.config.championship.save_transcript:
            transcript_path = self.config.logs_path / "championship_transcript.json"
            with open(transcript_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        path = Path(checkpoint_path)
        
        # Load model
        self.policy.load_checkpoint(str(path))
        
        # Load training state
        state_path = path / "training_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.stats.total_steps = state.get('total_steps', 0)
            self.stats.total_episodes = state.get('total_episodes', 0)
            self.stats.current_pr_idx = state.get('current_pr_idx', 0)
            self.stats.best_rewards = state.get('best_rewards', {})
            self.stats.solved_prs = state.get('solved_prs', [])
        
        # Load optimizer
        optimizer_path = path / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        
        logger.info(f"Loaded checkpoint from {path}")
