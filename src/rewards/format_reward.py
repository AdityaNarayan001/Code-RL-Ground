"""Format-aware reward shaping for agentic tool-use.

Provides dense reward signal during the critical cold-start phase of RL
training, when the model hasn't yet learned to produce valid tool calls.

This reward is GENERIC — it scores structured tool-call format adherence
regardless of the specific tool schema. Works with:
  - <tool>name(args)</tool>  (our native format)
  - ```tool\nname(args)\n```  (markdown-style)
  - Any bracketed structured action format

The reward is ADDITIVE and decays over training steps so it doesn't
interfere once the model has learned the format and is optimizing for
task-level reward.

Design principles:
  1. Reward ANY structural progress toward valid tool calls
  2. Give graduated signal (partial credit for partial format)
  3. Decay to zero so it doesn't distort late-stage training
  4. Be model-agnostic — no hardcoded token IDs
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class FormatRewardConfig:
    """Configuration for format reward shaping."""
    enabled: bool = True

    # --- reward magnitudes (added to the task reward) ---
    tool_tag_reward: float = 0.06       # Produced <tool>...</tool> tags
    valid_tool_name_reward: float = 0.04 # Used a real tool name
    valid_args_reward: float = 0.04      # Arguments parse correctly
    multi_step_reward: float = 0.03      # Produced >1 tool call across turns
    submit_reward: float = 0.03          # Called submit()

    # --- decay schedule ---
    warmup_steps: int = 50              # Full reward for first N steps
    decay_steps: int = 150              # Linear decay from warmup_steps to this
    # After decay_steps, format reward = 0 (model should be format-fluent)


# Precompiled patterns
_TOOL_TAG_PATTERN = re.compile(r'<tool>\s*(\w+)\s*\(([^)]*)\)\s*</tool>', re.DOTALL)
_TOOL_MARKDOWN_PATTERN = re.compile(r'```tool\s*\n\s*(\w+)\s*\(([^)]*)\)\s*\n\s*```', re.DOTALL)
_PARTIAL_OPEN_TAG = re.compile(r'<tool>', re.IGNORECASE)
_PARTIAL_CLOSE_TAG = re.compile(r'</tool>', re.IGNORECASE)
_FUNC_CALL_PATTERN = re.compile(r'\b(\w+)\s*\(', re.DOTALL)

# Known tool names for validation (generic set covering common agentic platforms)
DEFAULT_KNOWN_TOOLS = {
    # Our native tools
    "read_file", "write_file", "edit_file", "run_python", "submit",
    # Common agentic platform tools
    "search", "grep", "find", "ls", "cat", "bash", "execute",
    "create_file", "delete_file", "replace", "insert",
    "run_command", "terminal", "shell",
}


class FormatRewardScorer:
    """Score model outputs for structural format adherence.

    Returns a scalar reward ∈ [0, max_format_reward] that is added to
    the environment's task reward.  The scorer is stateless per-call;
    episode-level tracking (multi-step bonus) is handled by the caller.
    """

    def __init__(
        self,
        config: Optional[FormatRewardConfig] = None,
        known_tools: Optional[set] = None,
    ):
        self.config = config or FormatRewardConfig()
        self.known_tools = known_tools or DEFAULT_KNOWN_TOOLS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_turn(
        self,
        model_output: str,
        turn: int,
        total_tool_calls_so_far: int,
        training_step: int,
    ) -> Dict[str, float]:
        """Score a single model turn for format adherence.

        Args:
            model_output:  Raw text the model produced this turn.
            turn:          Which turn in the episode (1-indexed).
            total_tool_calls_so_far:  Cumulative valid tool calls in
                           this episode *before* this turn.
            training_step: Global training step (for decay).

        Returns:
            Dict with 'total' and per-component breakdown.
        """
        if not self.config.enabled:
            return {"total": 0.0}

        decay = self._decay_factor(training_step)
        if decay <= 0:
            return {"total": 0.0}

        breakdown: Dict[str, float] = {}

        # 1. Did the model produce <tool>...</tool> tags?
        full_matches = _TOOL_TAG_PATTERN.findall(model_output)
        if not full_matches:
            full_matches = _TOOL_MARKDOWN_PATTERN.findall(model_output)

        has_tool_tag = len(full_matches) > 0

        if has_tool_tag:
            breakdown["tool_tag"] = self.config.tool_tag_reward
        else:
            # Partial credit: produced at least an opening <tool> tag
            if _PARTIAL_OPEN_TAG.search(model_output):
                breakdown["tool_tag"] = self.config.tool_tag_reward * 0.3
            else:
                # Even more partial: produced something that looks like a function call
                if _FUNC_CALL_PATTERN.search(model_output):
                    breakdown["tool_tag"] = self.config.tool_tag_reward * 0.1
                else:
                    breakdown["tool_tag"] = 0.0

        # 2. Valid tool name?
        if full_matches:
            tool_name = full_matches[0][0]
            if tool_name.lower() in {t.lower() for t in self.known_tools}:
                breakdown["valid_tool_name"] = self.config.valid_tool_name_reward
            else:
                # Partial: it's a word that looks intentional (not gibberish)
                breakdown["valid_tool_name"] = self.config.valid_tool_name_reward * 0.3
        else:
            breakdown["valid_tool_name"] = 0.0

        # 3. Valid arguments?
        if full_matches:
            args_str = full_matches[0][1].strip()
            if self._args_parse_ok(args_str):
                breakdown["valid_args"] = self.config.valid_args_reward
            elif args_str:
                # Has something in the args, just doesn't parse perfectly
                breakdown["valid_args"] = self.config.valid_args_reward * 0.3
            else:
                # No args but that's okay for some tools (submit())
                tool_name = full_matches[0][0]
                if tool_name.lower() in {"submit", "ls"}:
                    breakdown["valid_args"] = self.config.valid_args_reward
                else:
                    breakdown["valid_args"] = 0.0
        else:
            breakdown["valid_args"] = 0.0

        # 4. Multi-step bonus (model is chaining tool calls across turns)
        if has_tool_tag and total_tool_calls_so_far >= 1:
            breakdown["multi_step"] = self.config.multi_step_reward
        else:
            breakdown["multi_step"] = 0.0

        # 5. Submit bonus (model learned to terminate episodes)
        if has_tool_tag:
            tool_name = full_matches[0][0]
            if tool_name.lower() == "submit":
                breakdown["submit"] = self.config.submit_reward
            else:
                breakdown["submit"] = 0.0
        else:
            # Check for any submit-like text
            if re.search(r'submit\s*\(\s*\)', model_output, re.IGNORECASE):
                breakdown["submit"] = self.config.submit_reward * 0.5
            else:
                breakdown["submit"] = 0.0

        # Apply decay
        raw_total = sum(breakdown.values())
        breakdown["total"] = raw_total * decay
        breakdown["decay_factor"] = decay

        return breakdown

    def score_episode(
        self,
        turn_outputs: List[str],
        training_step: int,
    ) -> float:
        """Score an entire episode's format adherence.

        Convenience method that sums per-turn format rewards.

        Args:
            turn_outputs:  List of model outputs per turn.
            training_step: Global training step.

        Returns:
            Total format reward for the episode.
        """
        total = 0.0
        tool_calls_so_far = 0

        for i, output in enumerate(turn_outputs):
            result = self.score_turn(
                model_output=output,
                turn=i + 1,
                total_tool_calls_so_far=tool_calls_so_far,
                training_step=training_step,
            )
            total += result["total"]

            # Count valid tool calls for multi-step tracking
            if _TOOL_TAG_PATTERN.search(output) or _TOOL_MARKDOWN_PATTERN.search(output):
                tool_calls_so_far += 1

        return total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decay_factor(self, step: int) -> float:
        """Compute decay factor for format reward.

        Returns 1.0 during warmup, linearly decays to 0.0 after decay_steps.
        """
        if step <= self.config.warmup_steps:
            return 1.0
        if step >= self.config.decay_steps:
            return 0.0
        # Linear decay
        progress = (step - self.config.warmup_steps) / (
            self.config.decay_steps - self.config.warmup_steps
        )
        return 1.0 - progress

    def _args_parse_ok(self, args_str: str) -> bool:
        """Check if argument string parses as key=value pairs."""
        if not args_str:
            return True  # No args is valid for submit(), etc.

        # Pattern: key="value" or key='value' or key=value
        arg_pattern = re.compile(
            r'(\w+)\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^,\s\)]+)'
        )
        matches = arg_pattern.findall(args_str)
        return len(matches) > 0
