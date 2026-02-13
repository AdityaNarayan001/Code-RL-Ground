#!/usr/bin/env python3
"""Evaluation script for testing trained models."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger
from src.data import PRLoader
from src.environment import CodeEnv
from src.environment.code_env import Action, ActionType, Observation
from src.agent import LLMPolicy
from src.utils.repo_state import RepoStateManager

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def evaluate_pr(
    policy: LLMPolicy,
    env: CodeEnv,
    pr_data: dict,
    dependency_prs: list,
    max_turns: int = 10
) -> dict:
    """Evaluate model on a single PR.
    
    Args:
        policy: The policy to evaluate
        env: The environment
        pr_data: PR task data dictionary
        dependency_prs: List of dependency PR IDs
        max_turns: Maximum turns allowed
        
    Returns:
        Evaluation results
    """
    logger = get_logger(__name__)
    
    obs = env.reset(pr_data, dependency_prs)
    turns = 0
    actions = []
    
    for turn in range(max_turns):
        # Generate action
        prompt = obs.content
        output = policy.generate(prompt)
        action_text = output.text
        actions.append(action_text)
        
        # Parse and execute
        action = env.parse_action(action_text)
        obs = env.step(action)
        turns += 1
        
        logger.info(f"  Turn {turn + 1}: terminal={obs.is_terminal}")
        
        if obs.is_terminal:
            break
    
    # Get episode result
    episode = env.get_episode()
    total_reward = episode.total_reward if episode else 0.0
    solved = episode.solved if episode else False
    
    return {
        'pr_id': pr_data['pr_id'],
        'solved': solved,
        'total_reward': total_reward,
        'turns': turns,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved model (LoRA weights)"
    )
    parser.add_argument(
        "--pr",
        type=str,
        default=None,
        help="Specific PR to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns per PR"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(log_dir=config.logs_path)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {args.model}")
    policy = LLMPolicy(config)
    policy.load()
    
    # Load LoRA weights
    lora_path = Path(args.model)
    if lora_path.exists():
        from peft import PeftModel
        policy.model = PeftModel.from_pretrained(policy.model, lora_path)
        logger.info("LoRA weights loaded")
    
    # Setup environment
    loader = PRLoader(config.dataset_path)
    repo_manager = RepoStateManager(
        base_repo_path=config.dataset_path / "base_repo",
        pr_data_path=config.dataset_path / "prs",
        cache_path=config.cache_path
    )
    env = CodeEnv(config, repo_manager)

    # Get PRs to evaluate
    if args.pr:
        pr_ids = [args.pr]
    else:
        tasks = loader.load_all()
        pr_ids = [t.pr_id for t in tasks]

    # Run evaluation
    results = []
    for pr_id in pr_ids:
        logger.info(f"\nEvaluating {pr_id}...")
        task = loader.load_pr(pr_id)
        result = evaluate_pr(policy, env, task.data, task.depends_on, args.max_turns)
        results.append(result)
        
        status = "✓ SOLVED" if result['solved'] else "✗ Not solved"
        logger.info(f"  {status} (reward={result['total_reward']:.3f})")

    # Summary
    solved = sum(1 for r in results if r['solved'])
    total = len(results)
    avg_reward = sum(r['total_reward'] for r in results) / total if total > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"  Solved: {solved}/{total} ({100*solved/total:.1f}%)")
    logger.info(f"  Average reward: {avg_reward:.3f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump({
                'model_path': args.model,
                'results': results,
                'summary': {
                    'solved': solved,
                    'total': total,
                    'solve_rate': solved / total if total > 0 else 0,
                    'avg_reward': avg_reward,
                }
            }, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
