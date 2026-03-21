"""Phase configuration for progressive RL training."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


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
    name: str = ""
    auto_submit_after_turn: Optional[int] = None
    system_prompt_override: Optional[str] = None


def load_phases_from_config(config) -> Dict[int, PhaseConfig]:
    """Load phase configs from the main config object.

    Falls back to DEFAULT_PHASES if phased_training section is missing.
    """
    try:
        raw = config._raw_config if hasattr(config, '_raw_config') else {}
        phased = raw.get('phased_training', {})
        phases_dict = phased.get('phases', {})

        if not phases_dict:
            return DEFAULT_PHASES

        result = {}
        for phase_num_str, phase_data in phases_dict.items():
            phase_num = int(phase_num_str)
            tools = phase_data.get('tools', [])
            pr_ids = phase_data.get('pr_ids', ['PR-001'])
            if isinstance(pr_ids, str) and pr_ids == 'all':
                pr_ids = ['all']

            result[phase_num] = PhaseConfig(
                phase=phase_num,
                name=phase_data.get('name', f'Phase {phase_num}'),
                max_turns=phase_data.get('max_turns', 1),
                available_tools=tools,
                group_size=phase_data.get('group_size', 4),
                advancement_threshold=phase_data.get('advancement_threshold', 0.8),
                advancement_window=phase_data.get('advancement_window', 10),
                advancement_required=phase_data.get('advancement_required', 7),
                pr_ids=pr_ids,
                auto_submit_after_turn=phase_data.get('auto_submit_after_turn'),
                system_prompt_override=phase_data.get('system_prompt_override'),
            )

        return result
    except Exception:
        return DEFAULT_PHASES


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
        advancement_threshold=0.50,
        advancement_window=10,
        advancement_required=5,
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
