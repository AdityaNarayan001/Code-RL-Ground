"""FastAPI server for the training dashboard."""

import os
# Disable tokenizer parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import collections
import math
import statistics
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import psutil

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


def _sanitize_floats(obj):
    """Recursively replace inf/nan floats with 0.0 so JSON serialization succeeds."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj

from .websocket import WebSocketManager
from ..utils.config import load_config, Config
from ..utils.logging import get_logger


logger = get_logger(__name__)


# Request/Response models
class TrainingStartRequest(BaseModel):
    config_overrides: Optional[Dict[str, Any]] = None


class TrainingStatus(BaseModel):
    is_running: bool
    current_step: int
    current_episode: int
    current_pr: Optional[str]
    solved_prs: List[str]
    avg_reward: float
    elapsed_time: float
    device: str = "auto"


class PRInfo(BaseModel):
    pr_id: str
    title: str
    description: str
    difficulty: int
    status: str
    best_reward: float
    attempts: int


# Global state
class ServerState:
    """Global server state."""
    def __init__(self):
        self.config: Optional[Config] = None
        self.training_task: Optional[asyncio.Task] = None
        self.training_thread: Optional[threading.Thread] = None
        self.stop_training_flag: bool = False
        self.is_training: bool = False
        self.training_stats: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.max_logs = 1000
        self.recent_logs: collections.deque = collections.deque(maxlen=1000)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.trainer = None  # Reference to active trainer for stop signaling
        # Incremental average tracking
        self._reward_sum: float = 0.0
        self._reward_count: int = 0
        # Recent rewards for stats
        self.recent_rewards: collections.deque = collections.deque(maxlen=100)
        # Step timing
        self._step_times: collections.deque = collections.deque(maxlen=100)
        self._last_step_time: Optional[float] = None
        # Gradient stats
        self._recent_grad_norms: collections.deque = collections.deque(maxlen=100)
        # Episode lengths
        self._episode_lengths: collections.deque = collections.deque(maxlen=100)


state = ServerState()
ws_manager = WebSocketManager()


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create FastAPI application.

    Args:
        config: Optional configuration

    Returns:
        FastAPI app
    """
    app = FastAPI(
        title="Code-RL-Ground",
        description="RL-based code learning environment",
        version="0.1.0"
    )

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load config
    if config:
        state.config = config
    else:
        state.config = load_config()

    # Track last broadcast index for log polling
    last_broadcast_idx = [0]

    # Background task to poll logs and broadcast to WebSocket clients
    async def log_broadcaster():
        """Poll logs from training thread and broadcast to clients."""
        while True:
            try:
                current_len = len(state.recent_logs)
                if current_len > last_broadcast_idx[0]:
                    # With deque, convert to list for safe slicing
                    logs_snapshot = list(state.recent_logs)
                    new_logs = logs_snapshot[last_broadcast_idx[0]:current_len]
                    for log in new_logs:
                        await ws_manager.broadcast(log)
                    last_broadcast_idx[0] = current_len
                # Reset index if deque has wrapped around
                if last_broadcast_idx[0] > current_len:
                    last_broadcast_idx[0] = 0
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(0.1)  # Poll every 100ms

    @app.on_event("startup")
    async def startup_event():
        """Start background tasks on app startup."""
        asyncio.create_task(log_broadcaster())

    # Routes
    @app.get("/")
    async def root():
        return {"status": "ok", "message": "Code-RL-Ground API"}

    @app.get("/api/config")
    async def get_config():
        """Get current configuration."""
        if not state.config:
            raise HTTPException(status_code=500, detail="Config not loaded")

        return {
            "project": {
                "name": state.config.project.name,
                "version": state.config.project.version
            },
            "model": {
                "name": state.config.model.name,
                "device": state.config.model.device
            },
            "training": {
                "algorithm": state.config.training.algorithm,
                "batch_size": state.config.training.batch_size,
                "learning_rate": state.config.training.learning_rate
            },
            "curriculum": {
                "strategy": state.config.curriculum.strategy,
                "solve_threshold": state.config.curriculum.solve_threshold
            }
        }

    @app.get("/api/status")
    async def get_status():
        """Get training status."""
        elapsed = 0.0
        if state.start_time:
            elapsed = (datetime.now() - state.start_time).total_seconds()

        # Detect device
        import torch
        if torch.cuda.is_available():
            device = "CUDA"
        elif torch.backends.mps.is_available():
            device = "MPS"
        else:
            device = "CPU"

        return TrainingStatus(
            is_running=state.is_training,
            current_step=state.training_stats.get('total_steps', 0),
            current_episode=state.training_stats.get('total_episodes', 0),
            current_pr=state.training_stats.get('current_pr_id'),
            solved_prs=state.training_stats.get('solved_prs', []),
            avg_reward=_sanitize_floats(state.training_stats.get('avg_reward', 0.0)),
            elapsed_time=elapsed,
            device=device
        )

    @app.get("/api/prs")
    async def get_prs():
        """Get all PR tasks with status."""
        from ..data import PRLoader

        loader = PRLoader(state.config.dataset_path)
        tasks = loader.load_all()

        solved = state.training_stats.get('solved_prs', [])
        best_rewards = state.training_stats.get('best_rewards', {})
        attempts_per_pr = state.training_stats.get('attempts_per_pr', {})

        result = []
        for task in tasks:
            status = "solved" if task.pr_id in solved else (
                "in_progress" if task.pr_id == state.training_stats.get('current_pr_id') else "pending"
            )
            result.append(PRInfo(
                pr_id=task.pr_id,
                title=task.title,
                description=task.description,
                difficulty=task.difficulty,
                status=status,
                best_reward=_sanitize_floats(best_rewards.get(task.pr_id, 0.0)),
                attempts=attempts_per_pr.get(task.pr_id, 0)
            ))

        return result

    @app.get("/api/pr/{pr_id}")
    async def get_pr(pr_id: str):
        """Get details for a specific PR."""
        from ..data import PRLoader

        loader = PRLoader(state.config.dataset_path)
        try:
            task = loader.load_pr(pr_id)
            return {
                "pr_id": task.pr_id,
                "title": task.title,
                "description": task.description,
                "difficulty": task.difficulty,
                "files_changed": task.files_changed,
                "depends_on": task.depends_on,
                "expected_changes": task.expected_changes,
                "test_cases": task.test_cases
            }
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"PR not found: {pr_id}")

    @app.post("/api/training/start")
    async def start_training(request: TrainingStartRequest):
        """Start training."""
        if state.is_training:
            raise HTTPException(status_code=400, detail="Training already in progress")

        # --- Validation ---
        # Check model path is accessible
        model_name = state.config.model.name
        model_path = Path(model_name)
        if model_path.exists():
            # Local path – verify it is a directory with model files
            if not model_path.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"Model path exists but is not a directory: {model_name}"
                )
        # If not a local path, assume it is a HuggingFace model id (validated at load time)

        # Check dataset path exists
        dataset_path = state.config.dataset_path
        if not dataset_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset path does not exist: {dataset_path}"
            )

        # Validate basic config sanity
        if state.config.training.learning_rate <= 0:
            raise HTTPException(
                status_code=400,
                detail="learning_rate must be positive"
            )
        if state.config.training.batch_size <= 0:
            raise HTTPException(
                status_code=400,
                detail="batch_size must be positive"
            )

        state.is_training = True
        state.start_time = datetime.now()
        state.training_stats = {}
        state.stop_training_flag = False
        # Reset incremental average trackers
        state._reward_sum = 0.0
        state._reward_count = 0
        state.recent_rewards.clear()
        state._step_times.clear()
        state._last_step_time = None
        state._recent_grad_norms.clear()
        state._episode_lengths.clear()

        # Run training in thread pool to avoid blocking
        def run_training_sync():
            from ..agent import LLMPolicy, GRPOTrainer
            from ..environment import CodeEnv
            from ..data import PRLoader, CurriculumManager
            from ..utils.repo_state import RepoStateManager

            loop = asyncio.new_event_loop()

            try:
                logger.info("Initializing training...")
                _append_log({'type': 'info', 'message': 'Initializing training...'})

                # Setup components
                loader = PRLoader(state.config.dataset_path)
                repo_manager = RepoStateManager(
                    base_repo_path=state.config.dataset_path / "base_repo",
                    pr_data_path=state.config.dataset_path / "prs",
                    cache_path=state.config.cache_path
                )

                curriculum = CurriculumManager(state.config, loader)
                env = CodeEnv(state.config, repo_manager)
                _append_log({'type': 'info', 'message': 'Environment ready'})

                logger.info("Loading model (this may take a few minutes)...")
                _append_log({'type': 'info', 'message': 'Loading model...'})
                policy = LLMPolicy(state.config)
                policy.load()
                _append_log({'type': 'info', 'message': 'Model loaded, setting up LoRA...'})
                policy.setup_lora()
                logger.info("Model loaded successfully!")
                _append_log({'type': 'info', 'message': 'LoRA ready!'})

                # Create GRPO trainer
                _append_log({'type': 'info', 'message': 'Using GRPO algorithm'})
                trainer = GRPOTrainer(
                    config=state.config,
                    policy=policy,
                    env=env,
                    pr_tasks=curriculum.get_remaining_tasks()
                )

                # Store reference for stop signaling
                state.trainer = trainer

                # Set callback for updates
                def thread_safe_broadcast(data: Dict[str, Any]):
                    _append_log(data)

                    # Update training stats from broadcasts
                    if data.get('type') == 'step':
                        now = time.time()
                        if state._last_step_time is not None:
                            state._step_times.append(now - state._last_step_time)
                        state._last_step_time = now

                        metrics = data.get('metrics', {})
                        state.training_stats['total_steps'] = data.get('step', 0)
                        state.training_stats['solve_rate'] = metrics.get('solve_rate', 0)
                        # Do NOT overwrite avg_reward from step events;
                        # the incremental average maintained from episodes is authoritative.
                        # Only seed it if we have no episode data yet.
                        if state._reward_count == 0:
                            state.training_stats['avg_reward'] = metrics.get(
                                'avg_reward', metrics.get('mean_reward', 0)
                            )
                        # Track gradient norms
                        grad_norm = metrics.get('grad_norm')
                        if grad_norm is not None:
                            state._recent_grad_norms.append(grad_norm)

                    elif data.get('type') == 'episode':
                        state.training_stats['total_episodes'] = data.get('episode', 0)
                        state.training_stats['current_pr_id'] = data.get('pr_id')

                        reward = data.get('reward', 0)
                        # Proper incremental average
                        state._reward_count += 1
                        state._reward_sum += reward
                        state.training_stats['avg_reward'] = state._reward_sum / state._reward_count
                        state.recent_rewards.append(reward)

                        # Track episode length
                        ep_len = data.get('turns', data.get('length'))
                        if ep_len is not None:
                            state._episode_lengths.append(ep_len)

                        # Update best_rewards
                        pr_id = data.get('pr_id')
                        if pr_id:
                            best_rewards = state.training_stats.setdefault('best_rewards', {})
                            if reward > best_rewards.get(pr_id, float('-inf')):
                                best_rewards[pr_id] = reward

                            # Update attempts_per_pr
                            attempts = state.training_stats.setdefault('attempts_per_pr', {})
                            attempts[pr_id] = attempts.get(pr_id, 0) + 1

                    elif data.get('type') == 'generation_token':
                        # Update current PR immediately when generation starts
                        if data.get('pr_id'):
                            state.training_stats['current_pr_id'] = data.get('pr_id')
                    elif data.get('type') == 'generation_complete':
                        # Update current group info
                        state.training_stats['current_group'] = data.get('group_idx', 0)
                        state.training_stats['current_pr_id'] = data.get('pr_id')
                    elif data.get('type') == 'pr_solved':
                        solved = state.training_stats.get('solved_prs', [])
                        if data.get('pr_id') not in solved:
                            solved.append(data.get('pr_id'))
                        state.training_stats['solved_prs'] = solved

                trainer.set_websocket_callback(thread_safe_broadcast)

                _append_log({'type': 'info', 'message': 'Starting training loop...'})

                # Run training
                result = trainer.train()

                state.training_stats = result.get('stats', {})
                _append_log({
                    'type': 'training_complete',
                    'result': result
                })

            except Exception as e:
                logger.error(f"Training error: {e}")
                logger.error(traceback.format_exc())
                _append_log({
                    'type': 'training_error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            finally:
                state.is_training = False
                state.trainer = None
                loop.close()

        # Start in thread pool
        state.training_thread = threading.Thread(target=run_training_sync, daemon=True)
        state.training_thread.start()

        return {"status": "started", "message": "Training started in background"}

    @app.post("/api/training/stop")
    async def stop_training():
        """Stop training."""
        if not state.is_training:
            raise HTTPException(status_code=400, detail="No training in progress")

        state.stop_training_flag = True

        # Signal the trainer to stop gracefully
        if state.trainer is not None:
            state.trainer.stop_requested = True

        state.is_training = False

        return {"status": "stopped", "message": "Stop signal sent"}

    @app.get("/api/logs")
    async def get_logs(limit: int = 100):
        """Get recent logs."""
        logs_list = list(state.recent_logs)
        return _sanitize_floats(logs_list[-limit:])

    @app.get("/api/metrics")
    async def get_metrics():
        """Get training metrics for charts."""
        # Extract metrics from logs
        rewards = []
        steps = []

        for log in list(state.recent_logs):
            if log.get('type') == 'step':
                metrics = log.get('metrics', {})
                steps.append({
                    'step': log.get('step', 0),
                    'avg_reward': metrics.get('avg_reward', metrics.get('mean_reward', 0)),
                    'solve_rate': metrics.get('solve_rate', 0),
                    'loss': metrics.get('loss', metrics.get('total_loss', 0)),
                    'pg_loss': metrics.get('pg_loss', 0),
                    'kl_loss': metrics.get('kl_loss', 0),
                    'grad_norm': metrics.get('grad_norm', 0),
                    'clip_frac': metrics.get('clip_frac', 0),
                    'max_reward': metrics.get('max_reward', 0)
                })
            elif log.get('type') == 'episode':
                rewards.append({
                    'episode': log.get('episode', 0),
                    'reward': log.get('reward', 0),
                    'pr_id': log.get('pr_id', ''),
                    'solved': log.get('solved', False)
                })

        return _sanitize_floats({
            'steps': steps[-100:],
            'episodes': rewards[-100:]
        })

    @app.get("/api/training/advanced-metrics")
    async def get_advanced_metrics():
        """Return advanced training metrics for detailed monitoring."""
        import torch

        # --- Policy entropy ---
        policy_entropy = state.training_stats.get('policy_entropy')

        # --- Reward statistics ---
        recent = list(state.recent_rewards)
        reward_std = 0.0
        if len(recent) >= 2:
            reward_std = statistics.stdev(recent)

        # Reward distribution histogram (10 buckets over last 100 rewards)
        reward_distribution: Dict[str, int] = {}
        if recent:
            min_r = min(recent)
            max_r = max(recent)
            num_buckets = 10
            bucket_width = (max_r - min_r) / num_buckets if max_r != min_r else 1.0
            for r in recent:
                bucket_idx = min(int((r - min_r) / bucket_width), num_buckets - 1)
                lo = round(min_r + bucket_idx * bucket_width, 3)
                hi = round(lo + bucket_width, 3)
                label = f"{lo}-{hi}"
                reward_distribution[label] = reward_distribution.get(label, 0) + 1

        # --- Curriculum progress ---
        curriculum_progress = {
            "solved_prs": state.training_stats.get('solved_prs', []),
            "current_pr": state.training_stats.get('current_pr_id'),
            "attempts_per_pr": state.training_stats.get('attempts_per_pr', {}),
        }

        # --- Memory usage ---
        mem = psutil.virtual_memory()
        memory_usage: Dict[str, Any] = {
            "system_total_mb": round(mem.total / (1024 ** 2), 1),
            "system_used_mb": round(mem.used / (1024 ** 2), 1),
            "system_percent": mem.percent,
        }
        if torch.cuda.is_available():
            memory_usage["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 1)
            memory_usage["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 1)

        # --- Step timing ---
        step_times_list = list(state._step_times)
        step_timing: Dict[str, float] = {}
        if step_times_list:
            step_timing["avg_seconds"] = round(statistics.mean(step_times_list), 4)
            step_timing["min_seconds"] = round(min(step_times_list), 4)
            step_timing["max_seconds"] = round(max(step_times_list), 4)

        # --- Gradient stats ---
        grad_list = list(state._recent_grad_norms)
        gradient_stats: Dict[str, float] = {}
        if grad_list:
            gradient_stats["mean"] = round(statistics.mean(grad_list), 6)
            gradient_stats["max"] = round(max(grad_list), 6)
            gradient_stats["min"] = round(min(grad_list), 6)

        # --- Episode length distribution ---
        ep_lens = list(state._episode_lengths)
        episode_length_distribution: Dict[str, float] = {}
        if ep_lens:
            episode_length_distribution["avg_turns"] = round(statistics.mean(ep_lens), 2)
            episode_length_distribution["min_turns"] = min(ep_lens)
            episode_length_distribution["max_turns"] = max(ep_lens)

        return _sanitize_floats({
            "policy_entropy": policy_entropy,
            "reward_std": reward_std,
            "reward_distribution": reward_distribution,
            "curriculum_progress": curriculum_progress,
            "memory_usage": memory_usage,
            "step_timing": step_timing,
            "gradient_stats": gradient_stats,
            "episode_length_distribution": episode_length_distribution,
        })

    @app.get("/api/training/checkpoints")
    async def list_checkpoints():
        """List available training checkpoints."""
        if not state.config:
            raise HTTPException(status_code=500, detail="Config not loaded")

        checkpoints_dir = state.config.checkpoints_path
        if not checkpoints_dir.exists():
            return {"checkpoints": []}

        checkpoints = []
        for entry in sorted(checkpoints_dir.iterdir()):
            if entry.is_dir():
                # Try to extract metadata
                meta = {"name": entry.name, "path": str(entry)}
                meta_file = entry / "trainer_state.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            trainer_state = json.load(f)
                        meta["step"] = trainer_state.get("global_step")
                        meta["epoch"] = trainer_state.get("epoch")
                    except Exception:
                        pass
                # Timestamp from directory modification time
                meta["modified"] = datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
                checkpoints.append(meta)

        return {"checkpoints": checkpoints}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive
                data = await websocket.receive_text()
                # Handle any client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)

    # Serve static files (React build) if available
    ui_path = state.config.resolve_path("./ui/build")
    if ui_path.exists():
        app.mount("/", StaticFiles(directory=str(ui_path), html=True), name="static")

    return app


def _append_log(entry: Dict[str, Any]):
    """Append a log entry with a timestamp to the recent_logs deque."""
    entry.setdefault('timestamp', datetime.now().isoformat())
    state.recent_logs.append(entry)


def run_server(host: str = "localhost", port: int = 8000, config_path: str = None):
    """Run the API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        config_path: Path to config file
    """
    config = load_config(config_path) if config_path else load_config()
    app = create_app(config)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
