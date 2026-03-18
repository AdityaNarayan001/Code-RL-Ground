"""Curriculum manager for progressive training."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque
import random

from .pr_loader import PRLoader, PRTask
from ..utils.config import Config, CurriculumConfig


@dataclass
class PRProgress:
    """Progress tracking for a single PR."""
    pr_id: str
    attempts: int = 0
    best_reward: float = 0.0
    consecutive_solves: int = 0
    solved: bool = False
    recent_rewards: deque = None
    recent_solves: deque = None

    def __post_init__(self):
        if self.recent_rewards is None:
            self.recent_rewards = deque(maxlen=20)
        if self.recent_solves is None:
            self.recent_solves = deque(maxlen=20)

    def update(self, reward: float, solved: bool):
        """Update progress with new attempt."""
        self.attempts += 1
        self.recent_rewards.append(reward)
        self.recent_solves.append(solved)

        if reward > self.best_reward:
            self.best_reward = reward

        if solved:
            self.consecutive_solves += 1
        else:
            self.consecutive_solves = 0

    def check_mastery(self, window: int = 5, required: int = 3) -> bool:
        """Check mastery using best M out of last K attempts.

        Args:
            window: Number of recent attempts to consider (K).
            required: Number of successes required within window (M).

        Returns:
            True if at least ``required`` of the last ``window`` attempts were solves.
        """
        recent = list(self.recent_solves)[-window:]
        return sum(recent) >= required

    @property
    def avg_reward(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    @property
    def success_rate(self) -> float:
        """Overall success rate across all attempts."""
        if not self.recent_solves:
            return 0.0
        total = len(self.recent_solves)
        if total == 0:
            return 0.0
        return sum(self.recent_solves) / total


class CurriculumManager:
    """Manage curriculum learning for PR tasks.
    
    Supports strategies:
    - dependency: Follow PR dependencies
    - difficulty: Sort by difficulty
    - random: Random order
    - sequential: Original order
    """
    
    def __init__(
        self,
        config: Config,
        pr_loader: PRLoader
    ):
        """Initialize curriculum manager.
        
        Args:
            config: Configuration object
            pr_loader: PR task loader
        """
        self.config = config
        self.curriculum_config = config.curriculum
        self.pr_loader = pr_loader
        
        # Load all tasks
        self.all_tasks = pr_loader.load_all()
        self.task_map = {t.pr_id: t for t in self.all_tasks}
        
        # Progress tracking
        self.progress: Dict[str, PRProgress] = {
            t.pr_id: PRProgress(pr_id=t.pr_id)
            for t in self.all_tasks
        }
        
        # Current order
        self._order: List[str] = []
        self._current_idx: int = 0
        
        # Initialize order based on strategy
        self._initialize_order()
    
    def _initialize_order(self):
        """Initialize PR order based on strategy."""
        strategy = self.curriculum_config.strategy
        
        if strategy == "dependency":
            self._order = self.pr_loader.get_topological_order()
        elif strategy == "difficulty":
            sorted_tasks = sorted(self.all_tasks, key=lambda t: t.difficulty)
            self._order = [t.pr_id for t in sorted_tasks]
        elif strategy == "random":
            self._order = [t.pr_id for t in self.all_tasks]
            random.shuffle(self._order)
            # Ensure dependencies are still respected
            if self.curriculum_config.respect_dependencies:
                self._order = self._ensure_dependency_order(self._order)
        else:  # sequential
            self._order = self.pr_loader.get_all_pr_ids()
        
        # Always respect dependencies if configured
        if self.curriculum_config.respect_dependencies:
            self._order = self._ensure_dependency_order(self._order)
    
    def _ensure_dependency_order(self, order: List[str]) -> List[str]:
        """Reorder to ensure dependencies come before dependents."""
        result = []
        added = set()
        
        def add_with_deps(pr_id: str):
            if pr_id in added:
                return
            
            # Add dependencies first
            deps = self.pr_loader.get_dependencies(pr_id)
            for dep in deps:
                add_with_deps(dep)
            
            if pr_id not in added:
                result.append(pr_id)
                added.add(pr_id)
        
        for pr_id in order:
            add_with_deps(pr_id)
        
        return result
    
    def get_current_task(self) -> Optional[PRTask]:
        """Get the current task to work on.
        
        Returns:
            Current PRTask or None if all done
        """
        if self._current_idx >= len(self._order):
            return None
        
        pr_id = self._order[self._current_idx]
        return self.task_map[pr_id]
    
    def get_remaining_tasks(self) -> List[PRTask]:
        """Get remaining tasks.
        
        Returns:
            List of remaining PRTasks
        """
        remaining = []
        for pr_id in self._order[self._current_idx:]:
            remaining.append(self.task_map[pr_id])
        return remaining
    
    def update_progress(self, pr_id: str, reward: float) -> bool:
        """Update progress for a PR.

        Uses a "best M out of last K" mastery check: the PR is considered
        solved when at least ``mastery_required`` of the last
        ``mastery_window`` attempts meet the solve threshold.  Falls back
        to the legacy ``min_consecutive_solves`` check when the windowed
        fields are not configured (both zero).

        Args:
            pr_id: PR ID
            reward: Reward from attempt

        Returns:
            True if PR is now solved
        """
        progress = self.progress[pr_id]
        solved = reward >= self.curriculum_config.solve_threshold
        progress.update(reward, solved)

        window = getattr(self.curriculum_config, 'mastery_window', 0)
        required = getattr(self.curriculum_config, 'mastery_required', 0)

        if window > 0 and required > 0:
            # Windowed mastery: best M out of last K
            if (progress.check_mastery(window=window, required=required) and
                    progress.best_reward >= self.curriculum_config.solve_threshold):
                progress.solved = True
                return True
        else:
            # Legacy: N consecutive solves
            if (progress.consecutive_solves >= self.curriculum_config.min_consecutive_solves and
                    progress.best_reward >= self.curriculum_config.solve_threshold):
                progress.solved = True
                return True

        return False
    
    def is_pr_solved(self, pr_id: str) -> bool:
        """Check if PR is solved.
        
        Args:
            pr_id: PR ID
            
        Returns:
            True if solved
        """
        return self.progress[pr_id].solved
    
    def advance(self) -> Optional[PRTask]:
        """Advance to next PR.
        
        Returns:
            Next PRTask or None if all done
        """
        if self._current_idx < len(self._order):
            current_pr = self._order[self._current_idx]
            self.progress[current_pr].solved = True
        
        self._current_idx += 1
        return self.get_current_task()
    
    def get_solved_prs(self) -> List[str]:
        """Get list of solved PR IDs.
        
        Returns:
            List of solved PR IDs
        """
        return [pr_id for pr_id, prog in self.progress.items() if prog.solved]
    
    def get_dependencies_for_current(self) -> List[str]:
        """Get dependency PR IDs for current task.
        
        Returns:
            List of dependency PR IDs (already solved)
        """
        current = self.get_current_task()
        if not current:
            return []
        
        return self.pr_loader.get_all_dependencies(current.pr_id)
    
    def is_complete(self) -> bool:
        """Check if curriculum is complete.
        
        Returns:
            True if all PRs are solved
        """
        return self._current_idx >= len(self._order)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics.
        
        Returns:
            Dictionary with curriculum stats
        """
        return {
            'total_prs': len(self.all_tasks),
            'completed_prs': len(self.get_solved_prs()),
            'current_pr_idx': self._current_idx,
            'current_pr': self._order[self._current_idx] if self._current_idx < len(self._order) else None,
            'order': self._order,
            'progress': {
                pr_id: {
                    'attempts': prog.attempts,
                    'best_reward': prog.best_reward,
                    'solved': prog.solved,
                    'avg_reward': prog.avg_reward
                }
                for pr_id, prog in self.progress.items()
            }
        }
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Return a summary of curriculum progress for monitoring.

        Returns:
            Dictionary with current PR, solved PRs, per-PR attempts and
            success rates, and overall completion percentage.
        """
        current_task = self.get_current_task()
        current_pr = current_task.pr_id if current_task else None

        per_pr = {}
        for pr_id, prog in self.progress.items():
            per_pr[pr_id] = {
                'attempts': prog.attempts,
                'best_reward': prog.best_reward,
                'avg_reward': prog.avg_reward,
                'success_rate': prog.success_rate,
                'solved': prog.solved,
                'consecutive_solves': prog.consecutive_solves,
            }

        solved_prs = self.get_solved_prs()
        total = len(self.all_tasks)

        return {
            'current_pr': current_pr,
            'current_pr_idx': self._current_idx,
            'total_prs': total,
            'solved_prs': solved_prs,
            'num_solved': len(solved_prs),
            'completion_pct': (len(solved_prs) / total * 100) if total > 0 else 0.0,
            'is_complete': self.is_complete(),
            'per_pr': per_pr,
        }

    def reset(self):
        """Reset curriculum to beginning."""
        self._current_idx = 0
        for progress in self.progress.values():
            progress.attempts = 0
            progress.best_reward = 0.0
            progress.consecutive_solves = 0
            progress.solved = False
            progress.recent_rewards.clear()
            progress.recent_solves.clear()
