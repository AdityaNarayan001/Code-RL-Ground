"""Phased training environments for progressive RL curriculum."""

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

from .code_env import Observation, Action, ActionType, Episode, CodeEnv
from ..utils.repo_state import RepoStateManager, RepoSnapshot
from ..utils.config import Config

logger = logging.getLogger(__name__)


class PhaseOneEnv:
    """Phase 1: Code completion. Model outputs raw Python code.

    Single turn, no tools. The model receives file content and a task
    description, and outputs the complete updated file.
    """

    def __init__(self, config: Config, repo_manager: Optional[RepoStateManager] = None):
        self.config = config
        self.env_config = config.environment
        self.repo_manager = repo_manager

        # Current state
        self.current_pr_data: Optional[Dict[str, Any]] = None
        self.current_state: Optional[RepoSnapshot] = None
        self.working_dir: Optional[Path] = None
        self.current_episode: Optional[Episode] = None

        # Lazy reward function
        self._reward_fn = None

    @property
    def reward_fn(self):
        """Lazy load reward function."""
        if self._reward_fn is None:
            from ..rewards import RewardFunction
            self._reward_fn = RewardFunction(self.config)
        return self._reward_fn

    def reset(self, pr_data: Dict[str, Any], dependency_prs: List[str]) -> Observation:
        """Reset environment for a new episode.

        Args:
            pr_data: PR task definition
            dependency_prs: List of dependency PRs (already solved)

        Returns:
            Initial observation with file content and task description
        """
        pr_id = pr_data['pr_id']
        self.current_pr_data = pr_data

        # Get repository state with dependencies applied
        self.current_state = self.repo_manager.get_state_for_pr(pr_id, dependency_prs)

        # Create working directory (resolve to absolute path)
        self.working_dir = self.repo_manager.create_working_directory(self.current_state)
        if not self.working_dir.is_absolute():
            self.working_dir = self.working_dir.resolve()

        # Read file content for files in pr_data['files_changed']
        files_changed = pr_data.get('files_changed', [])
        file_contents = []
        for file_path in files_changed:
            content = self.current_state.get_file(file_path)
            if content is not None:
                file_contents.append(f"### File: {file_path}\n```python\n{content}\n```")
            else:
                file_contents.append(f"### File: {file_path}\n(new file - does not exist yet)")

        # Build prompt
        prompt_parts = [
            f"# Task: {pr_data['title']}",
            "",
            f"## Description",
            pr_data['description'],
            "",
            "## Current File Content",
            "\n\n".join(file_contents),
            "",
            "Output ONLY the complete updated file content. "
            "Do not use any tools or special formatting. "
            "Do not wrap your output in markdown code fences.",
        ]
        prompt = "\n".join(prompt_parts)

        # Initialize episode
        self.current_episode = Episode(
            pr_id=pr_id,
            task_description=prompt,
        )

        return Observation(
            content=prompt,
            info={
                'pr_id': pr_id,
                'turn': 0,
                'files': files_changed,
            }
        )

    def step(self, action: Action) -> Observation:
        """Process model output as raw code.

        Args:
            action: Action where action.text contains the model's raw output

        Returns:
            Terminal Observation with reward
        """
        raw_text = action.text or action.raw_output or ""

        # Clean output: strip markdown fences if present
        cleaned = self._strip_markdown_fences(raw_text)

        # Determine target file (first file in files_changed)
        files_changed = self.current_pr_data.get('files_changed', [])
        if not files_changed:
            obs = Observation(
                content="No files to change.",
                is_terminal=True,
                reward=0.0,
                info={'error': 'no_files_changed'}
            )
            if self.current_episode:
                self.current_episode.turns.append((action, obs))
            return obs

        target_file = files_changed[0]

        # Write the code to the working directory
        full_path = self.working_dir / target_file
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(cleaned)

        # Get actual state from working directory
        actual_state = self.repo_manager.state_from_directory(
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
            actual_state=actual_state,
            expected_state=expected_state,
            pr_data=self.current_pr_data,
            working_dir=self.working_dir,
        )

        # Update episode
        if self.current_episode:
            self.current_episode.total_reward = reward_result.total
            self.current_episode.solved = reward_result.total >= self.config.curriculum.solve_threshold
            self.current_episode.submitted = True

        obs = Observation(
            content=f"Reward: {reward_result.total:.4f}",
            is_terminal=True,
            reward=reward_result.total,
            info={
                'turn': 1,
                'reward_breakdown': reward_result.breakdown,
                'solved': self.current_episode.solved if self.current_episode else False,
            }
        )

        if self.current_episode:
            self.current_episode.turns.append((action, obs))

        return obs

    def parse_action(self, text: str) -> Action:
        """Parse model output as a simple text action (no tool parsing in Phase 1)."""
        return Action(action_type=ActionType.SUBMIT, raw_output=text, text=text)

    def get_episode(self) -> Optional[Episode]:
        """Return current episode."""
        return self.current_episode

    def cleanup(self):
        """Clean up temporary working directory."""
        if self.working_dir and self.repo_manager:
            self.repo_manager.cleanup_work_dir(self.working_dir)
            self.working_dir = None

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences from text if present."""
        text = text.strip()
        # Match ```python ... ``` or ``` ... ```
        match = re.match(r'^```(?:python)?\s*\n(.*?)```\s*$', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class PhaseTwoEnv:
    """Phase 2: Tool format learning. Model wraps code in tool calls.

    Single turn. The model must output:
        <tool>write_file(path="...", content="...")</tool>

    Reward is split: 0.3 for correct format, 0.7 for code quality.
    """

    def __init__(self, config: Config, repo_manager: Optional[RepoStateManager] = None):
        self.config = config
        self.env_config = config.environment
        self.repo_manager = repo_manager

        # Current state
        self.current_pr_data: Optional[Dict[str, Any]] = None
        self.current_state: Optional[RepoSnapshot] = None
        self.working_dir: Optional[Path] = None
        self.current_episode: Optional[Episode] = None

        # Lazy reward function
        self._reward_fn = None

    @property
    def reward_fn(self):
        """Lazy load reward function."""
        if self._reward_fn is None:
            from ..rewards import RewardFunction
            self._reward_fn = RewardFunction(self.config)
        return self._reward_fn

    def reset(self, pr_data: Dict[str, Any], dependency_prs: List[str]) -> Observation:
        """Reset environment for a new episode.

        Args:
            pr_data: PR task definition
            dependency_prs: List of dependency PRs (already solved)

        Returns:
            Initial observation with file content and tool format instructions
        """
        pr_id = pr_data['pr_id']
        self.current_pr_data = pr_data

        # Get repository state with dependencies applied
        self.current_state = self.repo_manager.get_state_for_pr(pr_id, dependency_prs)

        # Create working directory (resolve to absolute path)
        self.working_dir = self.repo_manager.create_working_directory(self.current_state)
        if not self.working_dir.is_absolute():
            self.working_dir = self.working_dir.resolve()

        # Read file content for files in pr_data['files_changed']
        files_changed = pr_data.get('files_changed', [])
        file_contents = []
        for file_path in files_changed:
            content = self.current_state.get_file(file_path)
            if content is not None:
                file_contents.append(f"### File: {file_path}\n```python\n{content}\n```")
            else:
                file_contents.append(f"### File: {file_path}\n(new file - does not exist yet)")

        # Build prompt with tool format instructions
        prompt_parts = [
            f"# Task: {pr_data['title']}",
            "",
            f"## Description",
            pr_data['description'],
            "",
            "## Current File Content",
            "\n\n".join(file_contents),
            "",
            "Write the updated file using the following tool format:",
            '<tool>write_file(path="<filepath>", content="<full file content>")</tool>',
            "",
            "You MUST wrap your output in <tool>...</tool> tags using the write_file tool.",
        ]
        prompt = "\n".join(prompt_parts)

        # Initialize episode
        self.current_episode = Episode(
            pr_id=pr_id,
            task_description=prompt,
        )

        return Observation(
            content=prompt,
            info={
                'pr_id': pr_id,
                'turn': 0,
                'files': files_changed,
            }
        )

    def step(self, action: Action) -> Observation:
        """Process model output, scoring format and code quality.

        Args:
            action: Action with model output

        Returns:
            Terminal Observation with reward (format_reward + code_reward)
        """
        raw_text = action.text or action.raw_output or ""

        # --- Format sub-reward (0.30 total) ---
        format_reward = 0.0
        parsed_path = None
        parsed_content = None

        # Check for <tool>...</tool> tags
        has_tool_tags = bool(re.search(r'<tool>.*?</tool>', raw_text, re.DOTALL))
        if has_tool_tags:
            format_reward += 0.10

        # Parse tool call
        tool_pattern = r'<tool>\s*(\w+)\s*\((.*?)\)\s*</tool>'
        tool_match = re.search(tool_pattern, raw_text, re.DOTALL)

        if tool_match:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2)

            # Check tool name is write_file
            if tool_name == "write_file":
                format_reward += 0.10

            # Parse arguments
            arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|([^,\s\)]+))'
            parsed_args = {}
            for m in re.finditer(arg_pattern, args_str, re.DOTALL):
                key = m.group(1)
                value = m.group(2) if m.group(2) is not None else (
                    m.group(3) if m.group(3) is not None else m.group(4)
                )
                parsed_args[key] = value

            if 'path' in parsed_args and 'content' in parsed_args:
                format_reward += 0.10
                parsed_path = parsed_args['path']
                parsed_content = parsed_args['content']

        # --- Code sub-reward (scaled to 0.70) ---
        code_reward = 0.0

        if parsed_content is not None:
            # Determine target file
            files_changed = self.current_pr_data.get('files_changed', [])
            target_file = parsed_path or (files_changed[0] if files_changed else None)

            if target_file:
                # Write content to working directory
                full_path = self.working_dir / target_file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(parsed_content)

                # Get actual state
                actual_state = self.repo_manager.state_from_directory(
                    self.working_dir,
                    self.current_state.applied_prs + [self.current_pr_data['pr_id']]
                )

                # Get expected state
                expected_state = self.repo_manager.get_expected_state_after_pr(
                    self.current_pr_data['pr_id'],
                    self.current_state.applied_prs
                )

                # Compute code reward and scale to 0.7
                reward_result = self.reward_fn.compute(
                    actual_state=actual_state,
                    expected_state=expected_state,
                    pr_data=self.current_pr_data,
                    working_dir=self.working_dir,
                )
                code_reward = reward_result.total * 0.7

        total_reward = format_reward + code_reward

        # Update episode
        if self.current_episode:
            self.current_episode.total_reward = total_reward
            self.current_episode.solved = total_reward >= self.config.curriculum.solve_threshold
            self.current_episode.submitted = True

        obs = Observation(
            content=f"Reward: {total_reward:.4f} (format: {format_reward:.2f}, code: {code_reward:.4f})",
            is_terminal=True,
            reward=total_reward,
            info={
                'turn': 1,
                'format_reward': format_reward,
                'code_reward': code_reward,
                'has_tool_tags': has_tool_tags,
                'parsed_successfully': parsed_content is not None,
                'solved': self.current_episode.solved if self.current_episode else False,
            }
        )

        if self.current_episode:
            self.current_episode.turns.append((action, obs))

        return obs

    def parse_action(self, text: str) -> Action:
        """Parse model output — check for tool format."""
        return Action(action_type=ActionType.SUBMIT, raw_output=text, text=text)

    def get_episode(self) -> Optional[Episode]:
        """Return current episode."""
        return self.current_episode

    def cleanup(self):
        """Clean up temporary working directory."""
        if self.working_dir and self.repo_manager:
            self.repo_manager.cleanup_work_dir(self.working_dir)
            self.working_dir = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class PhaseThreeEnv:
    """Phase 3: Read then write. 2 turns with auto-submit.

    Turn 1: Model should read the target file.
    Turn 2: Model should write the updated version.
    After turn 2, submit is forced automatically.
    A read_bonus of 0.1 is added if the model reads a file on turn 1.
    """

    def __init__(self, config: Config, repo_manager: Optional[RepoStateManager] = None):
        self.config = config
        self.inner_env = CodeEnv(config=config, repo_state_manager=repo_manager)

        # Turn tracking
        self.turn_count: int = 0
        self.read_bonus: float = 0.0

        # Expose for trainer access
        self.current_pr_data: Optional[Dict[str, Any]] = None

    @property
    def reward_fn(self):
        """Delegate to inner env's reward function."""
        return self.inner_env.reward_fn

    def reset(self, pr_data: Dict[str, Any], dependency_prs: List[str]) -> Observation:
        """Reset environment for a new episode.

        Args:
            pr_data: PR task definition
            dependency_prs: List of dependency PRs (already solved)

        Returns:
            Initial observation with read-first instruction prepended
        """
        self.turn_count = 0
        self.read_bonus = 0.0
        self.current_pr_data = pr_data

        obs = self.inner_env.reset(pr_data, dependency_prs)

        # Prepend instruction to read first
        obs.content = (
            "Read the target file first, then write the updated version.\n\n"
            + obs.content
        )

        return obs

    def step(self, action: Action) -> Observation:
        """Process a turn.

        Turn 1: Delegate to inner env, check for read_file, return non-terminal.
        Turn 2: Delegate to inner env, force submit, add read_bonus.

        Args:
            action: Action from the model

        Returns:
            Observation (non-terminal after turn 1, terminal after turn 2)
        """
        self.turn_count += 1

        if self.turn_count == 1:
            # Turn 1: expect a read_file call
            obs = self.inner_env.step(action)

            # Check if it was a valid read_file call
            if (action.action_type == ActionType.TOOL_CALL
                    and action.tool_name == "read_file"):
                self.read_bonus = 0.1

            # Force non-terminal so we get a second turn
            obs.is_terminal = False
            return obs

        else:
            # Turn 2: delegate to inner env
            obs = self.inner_env.step(action)

            # Force submit if not already terminal
            if not obs.is_terminal:
                obs = self.inner_env._handle_submit()

            # Add read bonus
            obs.reward += self.read_bonus
            obs.is_terminal = True

            # Update episode info
            if self.inner_env.current_episode:
                self.inner_env.current_episode.total_reward = obs.reward

            obs.info['read_bonus'] = self.read_bonus
            return obs

    def parse_action(self, text: str) -> Action:
        """Delegate to inner env."""
        return self.inner_env.parse_action(text)

    def get_episode(self) -> Optional[Episode]:
        """Delegate to inner env."""
        return self.inner_env.current_episode

    def cleanup(self):
        """Delegate cleanup to inner env."""
        self.inner_env.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
