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
    """Phase 2: Structured output learning.

    Single turn. The model must output code wrapped in structured tags:

        <file path="pyutils/strings.py">
        ...complete file content...
        </file>

    This format is trivially parseable (no quote-escaping issues) and teaches
    the model to associate code with file paths using XML-style markup —
    a stepping stone toward full <tool>...</tool> syntax in Phase 3+.

    Reward: 0.3 for correct format, 0.7 for code quality.
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
        """Reset for new episode."""
        pr_id = pr_data['pr_id']
        self.current_pr_data = pr_data

        self.current_state = self.repo_manager.get_state_for_pr(pr_id, dependency_prs)
        self.working_dir = self.repo_manager.create_working_directory(self.current_state)
        if not self.working_dir.is_absolute():
            self.working_dir = self.working_dir.resolve()

        files_changed = pr_data.get('files_changed', [])
        file_contents = []
        for file_path in files_changed:
            content = self.current_state.get_file(file_path)
            if content is not None:
                file_contents.append(f"### File: {file_path}\n```python\n{content}\n```")
            else:
                file_contents.append(f"### File: {file_path}\n(new file)")

        # Example using the actual target file
        example_path = files_changed[0] if files_changed else "example.py"

        prompt_parts = [
            f"# Task: {pr_data['title']}",
            "",
            "## Description",
            pr_data['description'],
            "",
            "## Current File Content",
            "\n\n".join(file_contents),
            "",
            "## Output Format",
            "Wrap the complete updated file in <file> tags with the path attribute:",
            "",
            f'<file path="{example_path}">',
            "...complete updated file content here...",
            "</file>",
            "",
            "Output ONLY the <file> block. No explanations.",
        ]
        prompt = "\n".join(prompt_parts)

        self.current_episode = Episode(pr_id=pr_id, task_description=prompt)

        return Observation(
            content=prompt,
            info={'pr_id': pr_id, 'turn': 0, 'files': files_changed}
        )

    def step(self, action: Action) -> Observation:
        """Score format adherence and code quality."""
        raw_text = action.text or action.raw_output or ""

        # --- Format sub-reward (0.30 total) ---
        format_reward = 0.0
        parsed_path = None
        parsed_content = None

        # 1. Check for <file ...>...</file> tags (0.10)
        has_file_tags = bool(re.search(r'<file[\s>].*?</file>', raw_text, re.DOTALL))
        if has_file_tags:
            format_reward += 0.10

        # 2. Extract path attribute (0.10)
        path_match = re.search(r'<file\s+path\s*=\s*["\']([^"\']+)["\']', raw_text)
        if path_match:
            parsed_path = path_match.group(1)
            format_reward += 0.10

        # 3. Extract content between tags (0.10)
        content_match = re.search(
            r'<file[^>]*>\s*\n?(.*?)\s*</file>',
            raw_text, re.DOTALL
        )
        if content_match:
            parsed_content = content_match.group(1)
            format_reward += 0.10

        # --- Fallback extraction ---
        if parsed_content is None:
            # Try markdown fences
            code_block = re.search(r'```python\s*\n(.*?)```', raw_text, re.DOTALL)
            if code_block:
                parsed_content = code_block.group(1).strip()
            else:
                # Try raw code (if it looks like Python with def/class/import)
                stripped = raw_text.strip()
                if re.match(r'^(?:"""|\#|def |class |import |from )', stripped):
                    parsed_content = stripped

        # --- Code sub-reward (scaled to 0.70) ---
        code_reward = 0.0

        if parsed_content is not None:
            # Strip markdown fences if present inside
            parsed_content = PhaseOneEnv._strip_markdown_fences(parsed_content)

            files_changed = self.current_pr_data.get('files_changed', [])
            target_file = parsed_path or (files_changed[0] if files_changed else None)

            if target_file:
                full_path = self.working_dir / target_file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(parsed_content)

                actual_state = self.repo_manager.state_from_directory(
                    self.working_dir,
                    self.current_state.applied_prs + [self.current_pr_data['pr_id']]
                )
                expected_state = self.repo_manager.get_expected_state_after_pr(
                    self.current_pr_data['pr_id'],
                    self.current_state.applied_prs
                )

                reward_result = self.reward_fn.compute(
                    actual_state=actual_state,
                    expected_state=expected_state,
                    pr_data=self.current_pr_data,
                    working_dir=self.working_dir,
                )
                code_reward = reward_result.total * 0.7

        total_reward = format_reward + code_reward

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
                'has_file_tags': has_file_tags,
                'parsed_path': parsed_path,
                'parsed_content_length': len(parsed_content) if parsed_content else 0,
                'solved': self.current_episode.solved if self.current_episode else False,
            }
        )

        if self.current_episode:
            self.current_episode.turns.append((action, obs))

        return obs

    def parse_action(self, text: str) -> Action:
        """Parse model output."""
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

    Turn 1: <tool>read_file(path="...")</tool> — uses CodeEnv for tool execution
    Turn 2: <file path="...">...code...</file> — parsed directly (no quote escaping)

    After turn 2, the file content is written and reward is computed.
    A read_bonus of 0.1 is added if turn 1 is a valid read_file call.
    """

    def __init__(self, config: Config, repo_manager: Optional[RepoStateManager] = None):
        self.config = config
        self.repo_manager = repo_manager
        self.inner_env = CodeEnv(config=config, repo_state_manager=repo_manager)

        # Turn tracking
        self.turn_count: int = 0
        self.read_bonus: float = 0.0

        # Expose for trainer access
        self.current_pr_data: Optional[Dict[str, Any]] = None
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
        """Reset for new episode."""
        self.turn_count = 0
        self.read_bonus = 0.0
        self.current_pr_data = pr_data

        obs = self.inner_env.reset(pr_data, dependency_prs)
        self.current_episode = Episode(
            pr_id=pr_data['pr_id'],
            task_description=obs.content,
        )

        files_changed = pr_data.get('files_changed', [])
        target = files_changed[0] if files_changed else 'the target file'
        obs.content = (
            f"Complete this task in 2 steps:\n"
            f"Step 1: Read the file with <tool>read_file(path=\"{target}\")</tool>\n"
            f"Step 2: Write the complete updated file using:\n"
            f"<file path=\"{target}\">\n"
            f"...complete updated file content...\n"
            f"</file>\n\n"
            + obs.content
        )

        return obs

    def step(self, action: Action) -> Observation:
        """Process a turn.

        Turn 1: Delegate read_file to inner CodeEnv.
        Turn 2: Parse <file> tags directly, write to disk, compute reward.
        """
        self.turn_count += 1

        if self.turn_count == 1:
            # Turn 1: delegate to inner env for read_file execution
            obs = self.inner_env.step(action)

            # Bonus only when the read actually succeeded — otherwise the
            # model learns to "read" nonexistent paths for free reward
            if (action.action_type == ActionType.TOOL_CALL
                    and action.tool_name == "read_file"
                    and (obs.info or {}).get('success', False)):
                self.read_bonus = 0.1

            # Append instruction for Turn 2
            files_changed = self.current_pr_data.get('files_changed', [])
            target = files_changed[0] if files_changed else 'the file'
            obs.content += (
                f"\n\nNow write the complete updated file with your changes:\n"
                f"<file path=\"{target}\">\n"
                f"...complete updated file content here...\n"
                f"</file>"
            )

            obs.is_terminal = False
            if self.current_episode:
                self.current_episode.turns.append((action, obs))
            return obs

        else:
            # Turn 2: parse <file> tags directly (no CodeEnv write_file)
            raw_text = action.text or action.raw_output or ""

            # Extract content from <file path="...">...</file>
            parsed_content = None
            parsed_path = None

            # Try <file> tags first
            path_match = re.search(r'<file\s+path\s*=\s*["\']([^"\']+)["\']', raw_text)
            if path_match:
                parsed_path = path_match.group(1)

            content_match = re.search(
                r'<file[^>]*>\s*\n?(.*?)\s*</file>',
                raw_text, re.DOTALL
            )
            if content_match:
                parsed_content = content_match.group(1)

            # Fallback: try <tool>write_file(content="...")</tool>
            if parsed_content is None:
                wf_match = re.search(r'<tool>\s*write_file\s*\(', raw_text)
                if wf_match:
                    # Use the content parser from tools
                    inner_match = re.search(r'<tool>(.*?)</tool>', raw_text, re.DOTALL)
                    if inner_match:
                        inner = inner_match.group(1)
                        cm = re.search(r'content\s*=\s*["\']', inner)
                        if cm:
                            content = inner[cm.end():]
                            if content.rstrip().endswith('")') or content.rstrip().endswith("')"):
                                content = content.rstrip()[:-2]
                            elif content.endswith('"') or content.endswith("'"):
                                content = content[:-1]
                            content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                            parsed_content = content
                        pm = re.search(r'path\s*=\s*["\']([^"\']+)["\']', inner)
                        if pm:
                            parsed_path = pm.group(1)

            # Fallback: markdown fences
            if parsed_content is None:
                code_block = re.search(r'```python\s*\n(.*?)```', raw_text, re.DOTALL)
                if code_block:
                    parsed_content = code_block.group(1).strip()

            # Fallback: raw Python code
            if parsed_content is None:
                stripped = raw_text.strip()
                if re.match(r'^(?:"""|\#|def |class |import |from )', stripped):
                    parsed_content = stripped

            # Write to disk and compute reward
            working_dir = self.inner_env.working_dir
            files_changed = self.current_pr_data.get('files_changed', [])
            target_file = parsed_path or (files_changed[0] if files_changed else None)

            if parsed_content and target_file and working_dir:
                # Strip markdown fences if nested
                parsed_content = PhaseOneEnv._strip_markdown_fences(parsed_content)

                full_path = working_dir / target_file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(parsed_content)

                # Compute reward via reward_fn
                actual_state = self.repo_manager.state_from_directory(
                    working_dir,
                    self.inner_env.current_state.applied_prs + [self.current_pr_data['pr_id']]
                )
                expected_state = self.repo_manager.get_expected_state_after_pr(
                    self.current_pr_data['pr_id'],
                    self.inner_env.current_state.applied_prs
                )

                reward_result = self.reward_fn.compute(
                    actual_state=actual_state,
                    expected_state=expected_state,
                    pr_data=self.current_pr_data,
                    working_dir=working_dir,
                )
                reward = reward_result.total + self.read_bonus
                breakdown = reward_result.breakdown
            else:
                reward = self.read_bonus
                breakdown = {}

            if self.current_episode:
                self.current_episode.total_reward = reward
                self.current_episode.solved = reward >= self.config.curriculum.solve_threshold
                self.current_episode.submitted = True

            obs = Observation(
                content=f"Reward: {reward:.4f}",
                is_terminal=True,
                reward=reward,
                info={
                    'turn': 2,
                    'reward_breakdown': breakdown,
                    'solved': self.current_episode.solved if self.current_episode else False,
                    'read_bonus': self.read_bonus,
                    'parsed_path': parsed_path,
                    'parsed_content_length': len(parsed_content) if parsed_content else 0,
                }
            )

            if self.current_episode:
                self.current_episode.turns.append((action, obs))

            return obs

    def parse_action(self, text: str) -> Action:
        """Parse action — Turn 1 uses CodeEnv parser, Turn 2 is raw text."""
        if self.turn_count < 1:
            # Before Turn 1 or during Turn 1: parse for tool calls
            return self.inner_env.parse_action(text)
        else:
            # Turn 2: raw text (we parse <file> tags in step())
            return Action(action_type=ActionType.TEXT, raw_output=text, text=text)

    def get_episode(self) -> Optional[Episode]:
        """Return current episode."""
        return self.current_episode

    def cleanup(self):
        """Delegate cleanup to inner env."""
        self.inner_env.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
