"""Gym-like environment for code generation tasks."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy

from .sandbox import PythonSandbox
from .tools import ToolRegistry, ToolResult
from ..utils.repo_state import RepoStateManager, RepoSnapshot
from ..utils.config import Config, EnvironmentConfig


class ActionType(Enum):
    """Types of actions the agent can take."""
    TOOL_CALL = "tool_call"
    TEXT = "text"
    SUBMIT = "submit"


@dataclass
class Action:
    """An action taken by the agent."""
    action_type: ActionType
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    raw_output: str = ""


@dataclass
class Observation:
    """An observation returned to the agent."""
    content: str
    is_terminal: bool = False
    reward: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A single episode (attempt at solving a PR)."""
    pr_id: str
    task_description: str
    turns: List[Tuple[Action, Observation]] = field(default_factory=list)
    total_reward: float = 0.0
    solved: bool = False
    submitted: bool = False


class CodeEnv:
    """Environment for code generation with RL.
    
    Provides a Gym-like interface for training agents to solve
    coding tasks (PRs) using tools.
    """
    
    def __init__(
        self,
        config: Config,
        repo_state_manager: RepoStateManager
    ):
        """Initialize environment.
        
        Args:
            config: Configuration object
            repo_state_manager: Manager for repository state
        """
        self.config = config
        self.env_config = config.environment
        self.repo_manager = repo_state_manager
        
        # Initialize components
        self.sandbox = PythonSandbox(
            timeout=self.env_config.sandbox.timeout_per_execution,
            max_memory_mb=self.env_config.sandbox.max_memory_mb,
            allowed_imports=self.env_config.sandbox.allowed_imports
        )
        
        self.tools = ToolRegistry()
        
        # Current state
        self.current_episode: Optional[Episode] = None
        self.current_pr_data: Optional[Dict] = None
        self.current_state: Optional[RepoSnapshot] = None
        self.working_dir: Optional[Path] = None
        self.turn_count: int = 0
        
        # Import reward function (delayed to avoid circular import)
        self._reward_fn = None
    
    @property
    def reward_fn(self):
        """Lazy load reward function."""
        if self._reward_fn is None:
            from ..rewards import RewardFunction
            self._reward_fn = RewardFunction(self.config)
        return self._reward_fn
    
    def reset(
        self,
        pr_data: Dict[str, Any],
        dependency_prs: List[str]
    ) -> Observation:
        """Reset environment for a new episode.
        
        Args:
            pr_data: PR task definition
            dependency_prs: List of dependency PRs (already solved)
            
        Returns:
            Initial observation
        """
        pr_id = pr_data['pr_id']
        
        # Get repository state with dependencies applied
        self.current_state = self.repo_manager.get_state_for_pr(pr_id, dependency_prs)
        
        # Create working directory
        self.working_dir = self.repo_manager.create_working_directory(self.current_state)
        
        # Configure tools with working directory
        self.tools.set_working_dir(self.working_dir)
        
        # Store PR data
        self.current_pr_data = pr_data
        
        # Build task description
        task_description = self._build_task_prompt(pr_data)
        
        # Initialize episode
        self.current_episode = Episode(
            pr_id=pr_id,
            task_description=task_description
        )
        
        self.turn_count = 0
        
        # Return initial observation
        return Observation(
            content=task_description,
            info={
                'pr_id': pr_id,
                'turn': 0,
                'files': self.current_state.list_files()
            }
        )
    
    def _build_task_prompt(self, pr_data: Dict[str, Any]) -> str:
        """Build the task prompt for the agent."""
        prompt_parts = [
            f"# Task: {pr_data['title']}",
            "",
            f"## Description",
            pr_data['description'],
            "",
            f"## Files that may need changes",
            "\n".join(f"- {f}" for f in pr_data.get('files_changed', [])),
            "",
            "## Repository Structure",
            "\n".join(f"- {f}" for f in self.current_state.list_files()),
            "",
            self.tools.get_tools_prompt(),
            "",
            "Complete the task by using the tools. Call submit() when done."
        ]
        return "\n".join(prompt_parts)
    
    def step(self, action: Action) -> Observation:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Observation after the action
        """
        self.turn_count += 1
        
        # Check turn limit
        if self.env_config.mode == "multi_turn" and self.turn_count > self.env_config.max_turns:
            return self._create_terminal_observation(
                "Maximum turns reached. Episode ending.",
                force_submit=True
            )
        
        # Process action
        if action.action_type == ActionType.SUBMIT or action.tool_name == "submit":
            return self._handle_submit()
        
        elif action.action_type == ActionType.TOOL_CALL and action.tool_name:
            result = self.tools.execute(action.tool_name, **(action.tool_args or {}))
            
            # Check if this was a submit
            if result.data and result.data.get('action') == 'submit':
                return self._handle_submit()
            
            # Build observation
            obs_content = self._format_tool_result(action.tool_name, result)
            
            obs = Observation(
                content=obs_content,
                info={
                    'turn': self.turn_count,
                    'tool': action.tool_name,
                    'success': result.success
                }
            )
            
            # Record turn
            self.current_episode.turns.append((action, obs))
            
            return obs
        
        else:
            # Text-only response without tool call
            return Observation(
                content="Please use a tool to make progress. Available tools: " + 
                       ", ".join(self.tools.tools.keys()),
                info={'turn': self.turn_count}
            )
    
    def _format_tool_result(self, tool_name: str, result: ToolResult) -> str:
        """Format tool result for observation."""
        if result.success:
            return f"[{tool_name}] Success:\n{result.output}"
        else:
            return f"[{tool_name}] Error: {result.error}"
    
    def _handle_submit(self) -> Observation:
        """Handle submission and compute final reward."""
        self.current_episode.submitted = True
        
        # Get modifications
        modifications = self.tools.get_modifications()
        
        # Create state from modifications
        final_state = self.repo_manager.state_from_directory(
            self.working_dir,
            self.current_state.applied_prs + [self.current_pr_data['pr_id']]
        )
        
        # Get expected state
        expected_state = self.repo_manager.get_expected_state_after_pr(
            self.current_pr_data['pr_id'],
            self.current_state.applied_prs
        )
        
        # Compute reward
        reward_result = self.reward_fn.compute(
            actual_state=final_state,
            expected_state=expected_state,
            pr_data=self.current_pr_data,
            working_dir=self.working_dir
        )
        
        self.current_episode.total_reward = reward_result.total
        self.current_episode.solved = reward_result.total >= self.config.curriculum.solve_threshold
        
        # Build feedback
        feedback = self._build_feedback(reward_result)
        
        return Observation(
            content=feedback,
            is_terminal=True,
            reward=reward_result.total,
            info={
                'turn': self.turn_count,
                'reward_breakdown': reward_result.breakdown,
                'solved': self.current_episode.solved,
                'modifications': list(modifications.keys())
            }
        )
    
    def _build_feedback(self, reward_result) -> str:
        """Build feedback message based on config."""
        feedback_config = self.env_config.multi_turn_feedback
        parts = []
        
        if feedback_config.show_correct_incorrect:
            status = "CORRECT ✓" if self.current_episode.solved else "INCORRECT ✗"
            parts.append(f"Result: {status}")
        
        if feedback_config.show_partial_reward:
            parts.append(f"Reward: {reward_result.total:.4f}")
        
        if feedback_config.show_error_messages and reward_result.errors:
            parts.append("Errors:")
            for error in reward_result.errors[:3]:
                parts.append(f"  - {error}")
        
        if feedback_config.show_test_results and reward_result.test_results:
            tr = reward_result.test_results
            parts.append(f"Tests: {tr['passed']}/{tr['total']} passed")
        
        return "\n".join(parts)
    
    def _create_terminal_observation(
        self,
        message: str,
        force_submit: bool = False
    ) -> Observation:
        """Create a terminal observation."""
        if force_submit:
            return self._handle_submit()
        
        return Observation(
            content=message,
            is_terminal=True,
            reward=0.0
        )
    
    def parse_action(self, model_output: str) -> Action:
        """Parse model output into an action.
        
        Args:
            model_output: Raw model output text
            
        Returns:
            Parsed Action
        """
        # Try to parse tool call
        tool_call = self.tools.parse_tool_call(model_output)
        
        if tool_call:
            return Action(
                action_type=ActionType.TOOL_CALL if tool_call['tool'] != 'submit' else ActionType.SUBMIT,
                tool_name=tool_call['tool'],
                tool_args=tool_call['args'],
                raw_output=model_output
            )
        
        # Check for explicit submit keywords
        if any(kw in model_output.lower() for kw in ['submit()', '<submit>', '[[submit]]']):
            return Action(
                action_type=ActionType.SUBMIT,
                raw_output=model_output
            )
        
        # Just text
        return Action(
            action_type=ActionType.TEXT,
            text=model_output,
            raw_output=model_output
        )
    
    def get_episode(self) -> Optional[Episode]:
        """Get the current episode."""
        return self.current_episode
    
    def cleanup(self):
        """Clean up resources."""
        if self.working_dir:
            self.repo_manager.cleanup_work_dir(self.working_dir)
            self.working_dir = None
    
    def close(self):
        """Close the environment."""
        self.cleanup()
