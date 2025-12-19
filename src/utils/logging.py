"""Logging setup and utilities."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class TrainingLogger(logging.Logger):
    """Extended logger with training-specific methods."""
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self._websocket_callback = None
    
    def set_websocket_callback(self, callback):
        """Set callback for real-time WebSocket updates."""
        self._websocket_callback = callback
    
    def log_step(self, step: int, metrics: dict):
        """Log training step metrics."""
        self.info(f"Step {step}", extra={'extra_data': {'step': step, **metrics}})
        if self._websocket_callback:
            self._websocket_callback({
                'type': 'step',
                'step': step,
                'metrics': metrics
            })
    
    def log_episode(self, episode: int, pr_id: str, reward: float, solved: bool):
        """Log episode completion."""
        self.info(
            f"Episode {episode} | PR: {pr_id} | Reward: {reward:.4f} | Solved: {solved}",
            extra={'extra_data': {
                'episode': episode,
                'pr_id': pr_id,
                'reward': reward,
                'solved': solved
            }}
        )
        if self._websocket_callback:
            self._websocket_callback({
                'type': 'episode',
                'episode': episode,
                'pr_id': pr_id,
                'reward': reward,
                'solved': solved
            })
    
    def log_generation(self, pr_id: str, turn: int, content: str):
        """Log model generation for real-time display."""
        if self._websocket_callback:
            self._websocket_callback({
                'type': 'generation',
                'pr_id': pr_id,
                'turn': turn,
                'content': content
            })
    
    def log_pr_solved(self, pr_id: str, attempts: int, final_reward: float):
        """Log PR solved milestone."""
        self.info(
            f"🎉 PR SOLVED: {pr_id} | Attempts: {attempts} | Final Reward: {final_reward:.4f}",
            extra={'extra_data': {
                'event': 'pr_solved',
                'pr_id': pr_id,
                'attempts': attempts,
                'final_reward': final_reward
            }}
        )
        if self._websocket_callback:
            self._websocket_callback({
                'type': 'pr_solved',
                'pr_id': pr_id,
                'attempts': attempts,
                'final_reward': final_reward
            })
    
    def log_championship(self, results: dict):
        """Log championship round results."""
        self.info(
            f"🏆 CHAMPIONSHIP RESULTS: {results}",
            extra={'extra_data': {'event': 'championship', 'results': results}}
        )
        if self._websocket_callback:
            self._websocket_callback({
                'type': 'championship',
                'results': results
            })


# Replace default logger class
logging.setLoggerClass(TrainingLogger)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
    json_file: bool = True
) -> TrainingLogger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console: Whether to log to console
        json_file: Whether to log to JSON file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("code_rl_ground")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # JSON file handler
    if json_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.jsonl"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "code_rl_ground") -> TrainingLogger:
    """Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
