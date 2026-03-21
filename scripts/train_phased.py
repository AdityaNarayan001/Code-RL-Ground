#!/usr/bin/env python3
"""Phased RL training for teaching SLM to solve PR tasks.

Progressive skill learning in 5 phases:
  Phase 1: Code Completion - output correct Python (no tools)
  Phase 2: Tool Format - wrap code in <tool>write_file(...)</tool>
  Phase 3: Read Then Write - read_file -> write_file (2 turns)
  Phase 4: Full Tool Chain - read -> edit -> submit (5 turns)
  Phase 5: Full Curriculum - all PRs, all tools (10 turns)

Usage:
  python scripts/train_phased.py --config configs/config.yaml
  python scripts/train_phased.py --config configs/config.yaml --start-phase 3
  python scripts/train_phased.py --config configs/config.yaml --checkpoint checkpoints/phase_2/model
"""

import sys
from pathlib import Path
from collections import deque

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger
from src.data.pr_loader import PRLoader
from src.data.phase_config import DEFAULT_PHASES, PhaseConfig
from src.environment.code_env import CodeEnv
from src.agent.policy import LLMPolicy
from src.agent.grpo_trainer import GRPOTrainer
from src.utils.repo_state import RepoStateManager

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

# Human-readable phase names for logging
PHASE_NAMES = {
    1: "Code Completion",
    2: "Tool Format",
    3: "Read Then Write",
    4: "Full Tool Chain",
    5: "Full Curriculum",
}


def main():
    parser = argparse.ArgumentParser(description="Phased RL Training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to config file",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=1,
        help="Phase number to start from (1-5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load model weights from before training",
    )
    parser.add_argument(
        "--max-steps-per-phase",
        type=int,
        default=500,
        help="Maximum training steps per phase before forced advancement",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        log_dir=config.logs_path,
        level="DEBUG" if args.debug else "INFO",
    )
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Phased RL Training")
    logger.info("=" * 60)

    # Initialize model
    logger.info("Loading model...")
    policy = LLMPolicy(config)
    policy.load()
    policy.setup_lora()
    logger.info(f"Model loaded: {config.model.name}")
    logger.info(f"  Device: {config.model.device}")
    logger.info(f"  Quantization: {config.model.quantization}")

    # Load checkpoint weights if provided
    if args.checkpoint:
        policy.load_checkpoint(args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Initialize data
    loader = PRLoader(config.dataset_path)
    repo_manager = RepoStateManager(
        base_repo_path=config.dataset_path / "base_repo",
        pr_data_path=config.dataset_path / "prs",
        cache_path=config.cache_path,
    )

    # Phase loop
    for phase_num in range(args.start_phase, 6):
        phase_config = DEFAULT_PHASES[phase_num]
        phase_name = PHASE_NAMES.get(phase_num, f"Phase {phase_num}")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PHASE {phase_num}: {phase_name}")
        logger.info(f"  max_turns={phase_config.max_turns}  "
                     f"tools={phase_config.available_tools}  "
                     f"group_size={phase_config.group_size}")
        logger.info(f"  advancement: {phase_config.advancement_required}/"
                     f"{phase_config.advancement_window} >= "
                     f"{phase_config.advancement_threshold}")
        logger.info(f"  PR IDs: {phase_config.pr_ids}")
        logger.info(f"{'=' * 60}")

        # Override config for this phase
        config.environment.max_turns = phase_config.max_turns
        config.training.grpo.group_size = phase_config.group_size

        # Restrict available tools for this phase
        config.environment.tools.available = (
            phase_config.available_tools if phase_config.available_tools
            else ["submit"]  # Phase 1 has no tools but still needs submit
        )

        # Create environment
        env = CodeEnv(config, repo_manager)

        # Load PR tasks for this phase
        tasks = [loader.load_pr(pr_id) for pr_id in phase_config.pr_ids]
        logger.info(f"Loaded {len(tasks)} PR tasks for phase {phase_num}")

        # Create trainer for this phase
        trainer = GRPOTrainer(
            config=config,
            policy=policy,
            env=env,
            pr_tasks=tasks,
        )

        # Load previous phase checkpoint if it exists (not for the starting phase)
        prev_checkpoint = config.checkpoints_path / f"phase_{phase_num - 1}" / "model"
        if phase_num > args.start_phase and prev_checkpoint.exists():
            trainer.load_checkpoint(str(prev_checkpoint.parent))
            logger.info(f"Loaded Phase {phase_num - 1} checkpoint")

        # Training loop with advancement checking
        reward_history = deque(maxlen=phase_config.advancement_window)
        step = 0

        while step < args.max_steps_per_phase:
            # Use the trainer's internal task selection (it manages pr_tasks
            # index via stats.current_pr_idx). We grab the current task to
            # collect rollouts for it.
            pr_idx = trainer.stats.current_pr_idx
            if pr_idx >= len(tasks):
                # Wrap around to keep training on the phase's PR set
                trainer.stats.current_pr_idx = 0
                pr_idx = 0

            task = tasks[pr_idx]

            # Collect rollouts and do a GRPO update
            rollouts = trainer._collect_rollouts(task)
            if not rollouts:
                # No valid rollouts (e.g. env error) -- skip
                continue

            update_stats = trainer._grpo_update(rollouts)
            trainer._log_progress(rollouts, update_stats)
            step += 1

            # Track rewards for phase advancement
            for r in rollouts:
                reward_history.append(r.reward)

            # Check phase advancement criterion
            if len(reward_history) >= phase_config.advancement_required:
                met = sum(
                    1 for r in reward_history
                    if r >= phase_config.advancement_threshold
                )
                if met >= phase_config.advancement_required:
                    logger.info(
                        f"Phase {phase_num} COMPLETE! "
                        f"({met}/{len(reward_history)} above "
                        f"{phase_config.advancement_threshold})"
                    )
                    break

            # Broadcast phase progress (for UI if websocket callback is set)
            if trainer._ws_callback:
                met = sum(
                    1 for r in reward_history
                    if r >= phase_config.advancement_threshold
                )
                trainer._ws_callback({
                    'type': 'phase_progress',
                    'phase': phase_num,
                    'phase_name': phase_name,
                    'step': step,
                    'max_steps': args.max_steps_per_phase,
                    'recent_rewards': list(reward_history),
                    'threshold': phase_config.advancement_threshold,
                    'required': phase_config.advancement_required,
                    'met': met,
                })
        else:
            logger.info(
                f"Phase {phase_num} reached max steps ({args.max_steps_per_phase}) "
                f"without meeting advancement criteria. Advancing anyway."
            )

        # Save phase checkpoint
        checkpoint_dir = config.checkpoints_path / f"phase_{phase_num}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        policy.save(str(checkpoint_dir / "model"))
        logger.info(f"Saved Phase {phase_num} model to {checkpoint_dir / 'model'}")

        # Cleanup working directories
        env.cleanup()

    logger.info("=" * 60)
    logger.info("Phased training complete!")
    logger.info("=" * 60)

    # Save final model if configured
    if config.model_saving.save_final:
        final_path = Path(config.model_saving.final_model_path)
        logger.info(f"Saving final model to {final_path}")
        policy.save(str(final_path))
        logger.info("Final model saved!")


if __name__ == "__main__":
    main()
