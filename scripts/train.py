#!/usr/bin/env python3
"""Main training script for Code-RL-Ground."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger
from src.data import PRLoader, CurriculumManager
from src.environment import CodeEnv
from src.agent import LLMPolicy, PPOTrainer
from src.utils.repo_state import RepoStateManager

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def main():
    parser = argparse.ArgumentParser(description="Train the code RL agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(DEFAULT_CONFIG),
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(
        log_dir=config.logging.log_dir,
        level=logging.DEBUG if args.debug else logging.INFO
    )
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info(f"Starting {config.project.name} v{config.project.version}")
    logger.info("=" * 60)

    # Initialize components
    logger.info("Loading PR tasks...")
    loader = PRLoader(config.dataset_path)
    tasks = loader.load_all()
    logger.info(f"Loaded {len(tasks)} PR tasks")

    logger.info("Setting up repo state manager...")
    repo_manager = RepoStateManager(
        base_repo_path=config.dataset_path / "base_repo",
        pr_data_path=config.dataset_path / "prs",
        cache_path=config.cache_path
    )

    logger.info("Initializing curriculum...")
    curriculum = CurriculumManager(config, loader)
    ordered_tasks = curriculum.get_remaining_tasks()
    logger.info(f"Curriculum order: {[t.pr_id for t in ordered_tasks]}")

    logger.info("Creating environment...")
    env = CodeEnv(config, repo_manager)

    logger.info("Loading policy model...")
    policy = LLMPolicy(config)
    policy.load()
    policy.setup_lora()
    logger.info(f"Model loaded: {config.model.name}")
    logger.info(f"  Device: {config.model.device}")
    logger.info(f"  Quantization: {config.model.quantization}")

    # Select trainer based on algorithm
    algorithm = config.training.algorithm
    logger.info(f"Initializing {algorithm.upper()} trainer...")
    
    if algorithm == "grpo":
        from src.agent import GRPOTrainer
        trainer = GRPOTrainer(
            config=config,
            policy=policy,
            env=env,
            pr_tasks=ordered_tasks
        )
    else:
        trainer = PPOTrainer(
            config=config,
            policy=policy,
            env=env,
            pr_tasks=ordered_tasks
        )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run training
    logger.info("Starting training loop...")
    result = trainer.train()

    # Log final results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total steps: {result.get('total_steps', 0)}")
    logger.info(f"  Total episodes: {result.get('total_episodes', 0)}")
    logger.info(f"  Solved PRs: {result.get('solved_prs', [])}")
    logger.info(f"  Championship passed: {result.get('championship_passed', False)}")
    logger.info("=" * 60)

    # Save final model if configured
    if config.model_saving.save_final:
        final_path = Path(config.model_saving.final_model_path)
        logger.info(f"Saving final model to {final_path}")
        policy.save(final_path)
        logger.info("Final model saved!")

    return result


if __name__ == "__main__":
    main()
