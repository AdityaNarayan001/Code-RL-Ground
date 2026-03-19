"""Phase configuration for progressive RL training."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    phase: int
    max_turns: int
    available_tools: List[str]
    group_size: int
    advancement_threshold: float
    advancement_window: int  # K in "M of last K"
    advancement_required: int  # M
    pr_ids: List[str]
    auto_submit_after_turn: Optional[int] = None
    system_prompt_override: Optional[str] = None


DEFAULT_PHASES: Dict[int, PhaseConfig] = {
    1: PhaseConfig(
        phase=1,
        max_turns=1,
        available_tools=[],
        group_size=4,
        advancement_threshold=0.8,
        advancement_window=10,
        advancement_required=7,
        pr_ids=["PR-001", "PR-003"],
        auto_submit_after_turn=1,
        system_prompt_override=(
            "You are a code completion assistant. "
            "Given the current file and a task description, output ONLY the complete "
            "updated file content. Do not use any tools or special formatting."
        ),
    ),
    2: PhaseConfig(
        phase=2,
        max_turns=1,
        available_tools=["write_file"],
        group_size=4,
        advancement_threshold=0.75,
        advancement_window=10,
        advancement_required=7,
        pr_ids=["PR-001", "PR-003", "PR-005"],
        auto_submit_after_turn=1,
        system_prompt_override=(
            "You are a code editing assistant. "
            "Given the current file and a task description, output a tool call to "
            "write the updated file using this format:\n"
            '<tool>write_file(path="<filepath>", content="<full file content>")</tool>'
        ),
    ),
    3: PhaseConfig(
        phase=3,
        max_turns=2,
        available_tools=["read_file", "write_file"],
        group_size=4,
        advancement_threshold=0.7,
        advancement_window=12,
        advancement_required=8,
        pr_ids=["PR-001", "PR-003", "PR-005", "PR-006"],
        auto_submit_after_turn=2,
        system_prompt_override=(
            "You are a code editing assistant with file access. "
            "First, read the file you need to modify. Then, write the updated version. "
            "Use the tool format: <tool>tool_name(arg=\"value\")</tool>"
        ),
    ),
    4: PhaseConfig(
        phase=4,
        max_turns=5,
        available_tools=["read_file", "write_file", "edit_file", "list_directory", "submit"],
        group_size=4,
        advancement_threshold=0.7,
        advancement_window=15,
        advancement_required=10,
        pr_ids=[
            "PR-001", "PR-002", "PR-003", "PR-004",
            "PR-005", "PR-006", "PR-007", "PR-008",
        ],
        auto_submit_after_turn=None,
        system_prompt_override=None,
    ),
    5: PhaseConfig(
        phase=5,
        max_turns=10,
        available_tools=[
            "read_file", "write_file", "edit_file",
            "list_directory", "run_python", "search_code", "submit",
        ],
        group_size=4,
        advancement_threshold=0.9,
        advancement_window=20,
        advancement_required=16,
        pr_ids=[
            "PR-001", "PR-002", "PR-003", "PR-004", "PR-005",
            "PR-006", "PR-007", "PR-008", "PR-009", "PR-010",
        ],
        auto_submit_after_turn=None,
        system_prompt_override=None,
    ),
}
