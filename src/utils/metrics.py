"""Sophisticated logging and metrics tracking system.

Features:
- Structured JSON logging
- Metrics time series tracking
- Experiment run management
- CSV/JSON export
- Real-time WebSocket broadcasting
- Gradient and loss tracking
- Episode and step-level granularity
"""

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import threading
import numpy as np


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    step: int
    episode: int
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeLog:
    """Complete log of a single episode."""
    episode_id: int
    pr_id: str
    start_time: float
    end_time: float
    reward: float
    solved: bool
    num_turns: int
    actions: List[Dict[str, Any]]
    generation_texts: List[str]
    tool_calls: List[Dict[str, Any]]
    error: Optional[str] = None


@dataclass
class StepLog:
    """Log of a training step."""
    step: int
    timestamp: float
    loss: float
    pg_loss: float
    kl_loss: float
    entropy: float
    grad_norm: float
    learning_rate: float
    clip_frac: float
    mean_reward: float
    max_reward: float
    solve_rate: float
    num_episodes: int


class MetricsTracker:
    """Track and aggregate training metrics over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.running_stats: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    def log(self, name: str, value: float, step: int, episode: int = 0, **metadata):
        """Log a metric value."""
        with self._lock:
            point = MetricPoint(
                timestamp=time.time(),
                step=step,
                episode=episode,
                value=value,
                metadata=metadata
            )
            self.metrics[name].append(point)
            
            # Update running stats
            if name not in self.running_stats:
                self.running_stats[name] = {
                    'count': 0, 'sum': 0, 'min': float('inf'), 
                    'max': float('-inf'), 'last': 0
                }
            
            stats = self.running_stats[name]
            stats['count'] += 1
            stats['sum'] += value
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['last'] = value
    
    def get_recent(self, name: str, n: int = None) -> List[float]:
        """Get recent values for a metric."""
        n = n or self.window_size
        with self._lock:
            points = self.metrics.get(name, [])[-n:]
            return [p.value for p in points]
    
    def get_mean(self, name: str, n: int = None) -> float:
        """Get mean of recent values."""
        values = self.get_recent(name, n)
        return np.mean(values) if values else 0.0
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get running stats for a metric."""
        with self._lock:
            stats = self.running_stats.get(name, {})
            if not stats:
                return {'mean': 0, 'min': 0, 'max': 0, 'count': 0, 'last': 0}
            return {
                'mean': stats['sum'] / stats['count'] if stats['count'] > 0 else 0,
                'min': stats['min'] if stats['min'] != float('inf') else 0,
                'max': stats['max'] if stats['max'] != float('-inf') else 0,
                'count': stats['count'],
                'last': stats['last']
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def to_dataframe_dict(self) -> Dict[str, List]:
        """Export as dict suitable for pandas DataFrame."""
        with self._lock:
            result = defaultdict(list)
            for name, points in self.metrics.items():
                for p in points:
                    result['metric'].append(name)
                    result['timestamp'].append(p.timestamp)
                    result['step'].append(p.step)
                    result['episode'].append(p.episode)
                    result['value'].append(p.value)
            return dict(result)


class ExperimentLogger:
    """Complete experiment logging system."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: Path,
        config: Dict[str, Any] = None,
        console_level: str = "INFO",
        broadcast_callback: Callable = None,
        overwrite: bool = True  # Overwrite previous run by default
    ):
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use fixed "latest" dir if overwriting, otherwise timestamped
        if overwrite:
            self.log_dir = Path(log_dir) / experiment_name / "latest"
            # Clean up old files
            if self.log_dir.exists():
                import shutil
                shutil.rmtree(self.log_dir)
        else:
            self.log_dir = Path(log_dir) / experiment_name / self.run_id
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.start_time = time.time()
        self.broadcast_callback = broadcast_callback
        
        # Metrics tracker
        self.metrics = MetricsTracker()
        
        # Episode logs
        self.episodes: List[EpisodeLog] = []
        self.steps: List[StepLog] = []
        
        # Current episode state
        self._current_episode: Optional[Dict] = None
        
        # Setup Python logger
        self._setup_logger(console_level)
        
        # Save config
        self._save_config()
        
        # Open CSV files for streaming writes
        self._setup_csv_writers()
    
    def _setup_logger(self, level: str):
        """Setup Python logging."""
        self.logger = logging.getLogger(f"exp.{self.experiment_name}")
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()
        
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(console)
        
        # File handler (JSON)
        file_handler = logging.FileHandler(self.log_dir / "training.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        self.logger.addHandler(file_handler)
    
    def _setup_csv_writers(self):
        """Setup CSV writers for metrics."""
        # Metrics CSV
        self.metrics_csv = open(self.log_dir / "metrics.csv", 'w', newline='')
        self.metrics_writer = csv.writer(self.metrics_csv)
        self.metrics_writer.writerow([
            'timestamp', 'step', 'episode', 'metric', 'value'
        ])
        
        # Episodes CSV
        self.episodes_csv = open(self.log_dir / "episodes.csv", 'w', newline='')
        self.episodes_writer = csv.writer(self.episodes_csv)
        self.episodes_writer.writerow([
            'episode', 'pr_id', 'reward', 'solved', 'duration', 'num_turns'
        ])
        
        # Steps CSV
        self.steps_csv = open(self.log_dir / "steps.csv", 'w', newline='')
        self.steps_writer = csv.writer(self.steps_csv)
        self.steps_writer.writerow([
            'step', 'timestamp', 'loss', 'pg_loss', 'kl_loss', 'grad_norm',
            'mean_reward', 'max_reward', 'solve_rate', 'clip_frac'
        ])
    
    def _save_config(self):
        """Save experiment config."""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'run_id': self.run_id,
                'start_time': datetime.now().isoformat(),
                'config': self.config
            }, f, indent=2, default=str)
    
    def _broadcast(self, data: Dict[str, Any]):
        """Broadcast to WebSocket clients."""
        if self.broadcast_callback:
            self.broadcast_callback(data)
    
    # ================== Metric Logging ==================
    
    def log_metric(self, name: str, value: float, step: int, episode: int = 0):
        """Log a single metric."""
        self.metrics.log(name, value, step, episode)
        
        # Write to CSV
        self.metrics_writer.writerow([
            time.time(), step, episode, name, value
        ])
        self.metrics_csv.flush()
    
    def log_metrics(self, metrics: Dict[str, float], step: int, episode: int = 0):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step, episode)
    
    # ================== Step Logging ==================
    
    def log_step(
        self,
        step: int,
        loss: float,
        pg_loss: float = 0,
        kl_loss: float = 0,
        entropy: float = 0,
        grad_norm: float = 0,
        learning_rate: float = 0,
        clip_frac: float = 0,
        mean_reward: float = 0,
        max_reward: float = 0,
        solve_rate: float = 0,
        num_episodes: int = 0,
        extra: Dict[str, Any] = None
    ):
        """Log a training step."""
        step_log = StepLog(
            step=step,
            timestamp=time.time(),
            loss=loss,
            pg_loss=pg_loss,
            kl_loss=kl_loss,
            entropy=entropy,
            grad_norm=grad_norm,
            learning_rate=learning_rate,
            clip_frac=clip_frac,
            mean_reward=mean_reward,
            max_reward=max_reward,
            solve_rate=solve_rate,
            num_episodes=num_episodes
        )
        self.steps.append(step_log)
        
        # Log individual metrics
        self.log_metrics({
            'loss': loss,
            'pg_loss': pg_loss,
            'kl_loss': kl_loss,
            'entropy': entropy,
            'grad_norm': grad_norm,
            'clip_frac': clip_frac,
            'mean_reward': mean_reward,
            'max_reward': max_reward,
            'solve_rate': solve_rate
        }, step)
        
        # Write to CSV
        self.steps_writer.writerow([
            step, time.time(), loss, pg_loss, kl_loss, grad_norm,
            mean_reward, max_reward, solve_rate, clip_frac
        ])
        self.steps_csv.flush()
        
        # Console log (every 10 steps)
        if step % 10 == 0:
            self.logger.info(
                f"Step {step:5d} | Loss: {loss:.4f} | "
                f"Reward: {mean_reward:.3f} (max: {max_reward:.3f}) | "
                f"Solve: {solve_rate:.1%} | Grad: {grad_norm:.3f}"
            )
        
        # Broadcast
        self._broadcast({
            'type': 'step',
            'step': step,
            'metrics': {
                'loss': loss,
                'pg_loss': pg_loss,
                'kl_loss': kl_loss,
                'grad_norm': grad_norm,
                'mean_reward': mean_reward,
                'max_reward': max_reward,
                'solve_rate': solve_rate,
                'clip_frac': clip_frac,
                'avg_reward': self.metrics.get_mean('mean_reward', 100)
            }
        })
    
    # ================== Episode Logging ==================
    
    def start_episode(self, episode_id: int, pr_id: str):
        """Start tracking an episode."""
        self._current_episode = {
            'episode_id': episode_id,
            'pr_id': pr_id,
            'start_time': time.time(),
            'actions': [],
            'generation_texts': [],
            'tool_calls': []
        }
    
    def log_generation(self, text: str, turn: int):
        """Log a model generation within an episode."""
        if self._current_episode:
            self._current_episode['generation_texts'].append(text)
        
        self._broadcast({
            'type': 'generation_complete',
            'pr_id': self._current_episode['pr_id'] if self._current_episode else 'unknown',
            'turn': turn,
            'full_text': text
        })
    
    def log_tool_call(self, tool: str, args: Dict, result: Any, success: bool):
        """Log a tool call within an episode."""
        if self._current_episode:
            self._current_episode['tool_calls'].append({
                'tool': tool,
                'args': args,
                'result': str(result)[:500],  # Truncate large results
                'success': success,
                'timestamp': time.time()
            })
        
        self._broadcast({
            'type': 'tool_call',
            'tool': tool,
            'args': args,
            'success': success
        })
    
    def end_episode(self, reward: float, solved: bool, num_turns: int, error: str = None):
        """End tracking an episode."""
        if not self._current_episode:
            return
        
        episode_log = EpisodeLog(
            episode_id=self._current_episode['episode_id'],
            pr_id=self._current_episode['pr_id'],
            start_time=self._current_episode['start_time'],
            end_time=time.time(),
            reward=reward,
            solved=solved,
            num_turns=num_turns,
            actions=self._current_episode['actions'],
            generation_texts=self._current_episode['generation_texts'],
            tool_calls=self._current_episode['tool_calls'],
            error=error
        )
        self.episodes.append(episode_log)
        
        duration = episode_log.end_time - episode_log.start_time
        
        # Write to CSV
        self.episodes_writer.writerow([
            episode_log.episode_id,
            episode_log.pr_id,
            reward,
            solved,
            duration,
            num_turns
        ])
        self.episodes_csv.flush()
        
        # Log metrics
        self.log_metric('episode_reward', reward, 0, episode_log.episode_id)
        self.log_metric('episode_duration', duration, 0, episode_log.episode_id)
        self.log_metric('episode_solved', 1 if solved else 0, 0, episode_log.episode_id)
        
        # Console
        status = "✓ SOLVED" if solved else "✗"
        self.logger.info(
            f"Episode {episode_log.episode_id:4d} | {episode_log.pr_id} | "
            f"R={reward:.3f} | {status} | {duration:.1f}s"
        )
        
        # Broadcast
        self._broadcast({
            'type': 'episode',
            'episode': episode_log.episode_id,
            'pr_id': episode_log.pr_id,
            'reward': reward,
            'solved': solved,
            'duration': duration,
            'message': f'Episode {episode_log.episode_id}: {episode_log.pr_id} - R={reward:.2f}'
        })
        
        self._current_episode = None
    
    # ================== Milestone Logging ==================
    
    def log_pr_solved(self, pr_id: str, attempts: int, best_reward: float):
        """Log when a PR is solved."""
        self.logger.info(f"🎉 PR SOLVED: {pr_id} | Attempts: {attempts} | Best: {best_reward:.3f}")
        
        self._broadcast({
            'type': 'pr_solved',
            'pr_id': pr_id,
            'attempts': attempts,
            'best_reward': best_reward
        })
        
        # Save milestone
        milestone = {
            'event': 'pr_solved',
            'pr_id': pr_id,
            'attempts': attempts,
            'best_reward': best_reward,
            'timestamp': datetime.now().isoformat(),
            'elapsed': time.time() - self.start_time
        }
        with open(self.log_dir / "milestones.jsonl", 'a') as f:
            f.write(json.dumps(milestone) + '\n')
    
    def log_checkpoint(self, path: str, step: int):
        """Log checkpoint save."""
        self.logger.info(f"💾 Checkpoint saved: {path} (step {step})")
        
        self._broadcast({
            'type': 'checkpoint',
            'path': path,
            'step': step
        })
    
    def log_training_complete(self, stats: Dict[str, Any]):
        """Log training completion."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"🏁 Training complete! Elapsed: {elapsed/3600:.2f}h")
        
        # Save final summary
        summary = {
            'experiment_name': self.experiment_name,
            'run_id': self.run_id,
            'elapsed_seconds': elapsed,
            'total_steps': len(self.steps),
            'total_episodes': len(self.episodes),
            'final_stats': stats,
            'metric_summary': self.metrics.get_all_stats()
        }
        with open(self.log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._broadcast({
            'type': 'training_complete',
            'result': stats
        })
    
    def log_error(self, error: str, traceback: str = None):
        """Log an error."""
        self.logger.error(f"❌ Error: {error}")
        if traceback:
            self.logger.error(traceback)
        
        self._broadcast({
            'type': 'training_error',
            'error': error,
            'traceback': traceback
        })
    
    # ================== Export & Cleanup ==================
    
    def export_metrics_json(self) -> Path:
        """Export all metrics to JSON."""
        path = self.log_dir / "metrics_export.json"
        with open(path, 'w') as f:
            json.dump(self.metrics.to_dataframe_dict(), f)
        return path
    
    def export_episodes_json(self) -> Path:
        """Export all episodes to JSON."""
        path = self.log_dir / "episodes_export.json"
        with open(path, 'w') as f:
            json.dump([asdict(e) for e in self.episodes], f, default=str)
        return path
    
    def close(self):
        """Close file handles."""
        self.metrics_csv.close()
        self.episodes_csv.close()
        self.steps_csv.close()
        
        # Export final data
        self.export_metrics_json()
        self.export_episodes_json()
        
        self.logger.info(f"📁 Logs saved to: {self.log_dir}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.log_error(str(exc_val), exc_tb)
        self.close()


# ================== Convenience Functions ==================

def create_experiment_logger(
    config,
    broadcast_callback: Callable = None
) -> ExperimentLogger:
    """Create an experiment logger from config."""
    return ExperimentLogger(
        experiment_name=config.project.name,
        log_dir=config.logs_path,
        config=config.__dict__ if hasattr(config, '__dict__') else {},
        broadcast_callback=broadcast_callback
    )


# Keep backward compatibility
def get_logger(name: str) -> logging.Logger:
    """Get a standard Python logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class TrainingLogger:
    """Legacy training logger for backward compatibility."""
    
    def __init__(self, config):
        self.config = config
        self._ws_callback = None
        self.logger = get_logger("training")
    
    def set_websocket_callback(self, callback):
        self._ws_callback = callback
