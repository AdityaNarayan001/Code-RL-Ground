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
    phased: bool = False
    start_phase: int = 1
    resume: bool = False


class TrainingStatus(BaseModel):
    is_running: bool
    current_step: int
    current_episode: int
    current_pr: Optional[str]
    solved_prs: List[str]
    avg_reward: float
    elapsed_time: float
    device: str = "auto"
    current_phase: int = 0
    phase_name: str = ""
    phase_progress: Optional[Dict[str, Any]] = None
    is_stopping: bool = False


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
        # Phase tracking
        self.current_phase: int = 0  # 0 = regular, 1-5 = phased
        self.phase_name: str = ""
        self.phase_reward_history: collections.deque = collections.deque(maxlen=10)
        # Coarse lock guarding multi-field state snapshots and mutations
        self.lock = threading.RLock()
        # Monotonic sequence number for log entries (for broadcast tracking)
        self.log_seq: int = 0
        # Lazily loaded phase config cache (invalidated when training starts)
        self.phases_cache: Optional[Dict[int, Any]] = None


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

    # Track last broadcast log sequence number for log polling
    last_broadcast_seq = [0]

    # Background task to poll logs and broadcast to WebSocket clients
    async def log_broadcaster():
        """Poll logs from training thread and broadcast to clients."""
        while True:
            try:
                logs_snapshot = list(state.recent_logs)
                for log in logs_snapshot:
                    seq = log.get('_seq', 0)
                    if seq > last_broadcast_seq[0]:
                        await ws_manager.broadcast(log)
                        last_broadcast_seq[0] = seq
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

        # Take a consistent snapshot of mutable state
        with state.lock:
            is_running = state.is_training
            stop_flag = state.stop_training_flag
            stats = dict(state.training_stats)
            current_phase = state.current_phase
            phase_name = state.phase_name
            rewards_list = list(state.phase_reward_history)

        # Stopping = stop requested but the training thread is still winding down
        is_stopping = bool(
            stop_flag and is_running
            and state.training_thread is not None
            and state.training_thread.is_alive()
        )

        # Build phase progress info
        phase_progress = None
        if current_phase > 0:
            # Lazily load and cache phase config (invalidated on training start)
            try:
                if state.phases_cache is None:
                    from ..data.phase_config import load_phases_from_config
                    state.phases_cache = load_phases_from_config(state.config)
                pcfg = state.phases_cache.get(current_phase)
                if pcfg:
                    met = sum(1 for r in rewards_list if r >= pcfg.advancement_threshold)
                    phase_progress = {
                        'recent_rewards': [_sanitize_floats(r) for r in rewards_list],
                        'threshold': pcfg.advancement_threshold,
                        'required': pcfg.advancement_required,
                        'met': met,
                        'window': pcfg.advancement_window
                    }
            except Exception:
                pass

        return TrainingStatus(
            is_running=is_running,
            current_step=stats.get('total_steps', 0),
            current_episode=stats.get('total_episodes', 0),
            current_pr=stats.get('current_pr_id'),
            solved_prs=stats.get('solved_prs', []),
            avg_reward=_sanitize_floats(stats.get('avg_reward', 0.0)),
            elapsed_time=elapsed,
            device=device,
            current_phase=current_phase,
            phase_name=phase_name,
            phase_progress=phase_progress,
            is_stopping=is_stopping
        )

    @app.get("/api/prs")
    async def get_prs():
        """Get all PR tasks with status."""
        from ..data import PRLoader

        loader = PRLoader(state.config.dataset_path)
        tasks = loader.load_all()

        with state.lock:
            solved = list(state.training_stats.get('solved_prs', []))
            best_rewards = dict(state.training_stats.get('best_rewards', {}))
            attempts_per_pr = dict(state.training_stats.get('attempts_per_pr', {}))
            current_pr_id = state.training_stats.get('current_pr_id')

        result = []
        for task in tasks:
            status = "solved" if task.pr_id in solved else (
                "in_progress" if task.pr_id == current_pr_id else "pending"
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

        with state.lock:
            state.is_training = True
            state.start_time = datetime.now()
            state.training_stats = {}
            state.stop_training_flag = False
            # Invalidate cached phase config so it is reloaded for this run
            state.phases_cache = None
            # Reset incremental average trackers
            state._reward_sum = 0.0
            state._reward_count = 0
            state.recent_rewards.clear()
            state._step_times.clear()
            state._last_step_time = None
            state._recent_grad_norms.clear()
            state._episode_lengths.clear()

        is_phased = request.phased
        start_phase = request.start_phase
        is_resume = request.resume

        # Run training in thread pool to avoid blocking
        def run_training_sync():
            from ..agent import LLMPolicy, GRPOTrainer
            from ..environment import CodeEnv
            from ..data import PRLoader, CurriculumManager
            from ..utils.repo_state import RepoStateManager

            nonlocal start_phase
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

                logger.info("Loading model (this may take a few minutes)...")
                _append_log({'type': 'info', 'message': 'Loading model...'})
                policy = LLMPolicy(state.config)
                policy.load()
                _append_log({'type': 'info', 'message': 'Model loaded, setting up LoRA...'})
                policy.setup_lora()
                logger.info("Model loaded successfully!")
                _append_log({'type': 'info', 'message': 'LoRA ready!'})

                # Handle resume: scan checkpoints and load latest weights
                if is_resume and is_phased:
                    checkpoints_dir = state.config.checkpoints_path
                    completed = []
                    if checkpoints_dir.exists():
                        for entry in sorted(checkpoints_dir.iterdir()):
                            if entry.is_dir() and entry.name.startswith("phase_"):
                                try:
                                    pnum = int(entry.name.split("_")[1])
                                except (IndexError, ValueError):
                                    continue
                                model_dir = entry / "model"
                                if model_dir.exists() and model_dir.is_dir():
                                    completed.append(pnum)
                    completed.sort()
                    if completed:
                        latest_phase = max(completed)
                        resume_checkpoint = str(checkpoints_dir / f"phase_{latest_phase}" / "model")
                        start_phase = latest_phase + 1
                        _append_log({'type': 'info', 'message': f'Resuming from Phase {latest_phase} checkpoint'})
                        logger.info(f"Resuming from Phase {latest_phase} checkpoint: {resume_checkpoint}")
                        policy.load_checkpoint(resume_checkpoint)
                        _append_log({'type': 'info', 'message': f'Loaded checkpoint weights from {resume_checkpoint}'})
                    else:
                        _append_log({'type': 'info', 'message': 'No checkpoints found, starting from Phase 1'})
                        logger.info("Resume requested but no checkpoints found, starting fresh")

                # Set callback for updates
                def thread_safe_broadcast(data: Dict[str, Any]):
                    _append_log(data)

                    # Update training stats from broadcasts (under lock for consistent snapshots)
                    with state.lock:
                        if data.get('type') == 'step':
                            now = time.time()
                            if state._last_step_time is not None:
                                state._step_times.append(now - state._last_step_time)
                            state._last_step_time = now

                            metrics = data.get('metrics', {})
                            state.training_stats['total_steps'] = data.get('step', 0)
                            state.training_stats['solve_rate'] = metrics.get('solve_rate', 0)
                            if state._reward_count == 0:
                                state.training_stats['avg_reward'] = metrics.get(
                                    'avg_reward', metrics.get('mean_reward', 0)
                                )
                            grad_norm = metrics.get('grad_norm')
                            if grad_norm is not None:
                                state._recent_grad_norms.append(grad_norm)

                        elif data.get('type') == 'episode':
                            state.training_stats['total_episodes'] = data.get('episode', 0)
                            state.training_stats['current_pr_id'] = data.get('pr_id')

                            reward = data.get('reward', 0)
                            state._reward_count += 1
                            state._reward_sum += reward
                            state.training_stats['avg_reward'] = state._reward_sum / state._reward_count
                            state.recent_rewards.append(reward)
                            state.phase_reward_history.append(reward)

                            ep_len = data.get('turns', data.get('length'))
                            if ep_len is not None:
                                state._episode_lengths.append(ep_len)

                            pr_id = data.get('pr_id')
                            if pr_id:
                                best_rewards = state.training_stats.setdefault('best_rewards', {})
                                if reward > best_rewards.get(pr_id, float('-inf')):
                                    best_rewards[pr_id] = reward
                                attempts = state.training_stats.setdefault('attempts_per_pr', {})
                                attempts[pr_id] = attempts.get(pr_id, 0) + 1

                        elif data.get('type') == 'phase_change':
                            state.current_phase = data.get('phase', 0)
                            state.phase_name = data.get('name', '')
                            state.phase_reward_history.clear()

                        elif data.get('type') == 'generation_token':
                            if data.get('pr_id'):
                                state.training_stats['current_pr_id'] = data.get('pr_id')
                        elif data.get('type') == 'generation_complete':
                            state.training_stats['current_group'] = data.get('group_idx', 0)
                            state.training_stats['current_pr_id'] = data.get('pr_id')
                        elif data.get('type') == 'pr_solved':
                            solved = state.training_stats.get('solved_prs', [])
                            if data.get('pr_id') not in solved:
                                solved.append(data.get('pr_id'))
                            state.training_stats['solved_prs'] = solved

                if is_phased:
                    # ---- PHASED TRAINING ----
                    from ..environment.phase_env import PhaseOneEnv, PhaseTwoEnv, PhaseThreeEnv
                    from ..data.phase_config import DEFAULT_PHASES, load_phases_from_config
                    import shutil as _shutil

                    PHASES = load_phases_from_config(state.config)

                    # Fresh phased training: clean up old checkpoints
                    if not is_resume:
                        old_ckpt_dir = state.config.checkpoints_path
                        if old_ckpt_dir.exists():
                            for entry in old_ckpt_dir.iterdir():
                                if entry.is_dir() and entry.name.startswith("phase_"):
                                    _shutil.rmtree(entry, ignore_errors=True)
                            _append_log({'type': 'info', 'message': 'Cleared previous phase checkpoints'})

                    for phase_num in sorted(PHASES.keys()):
                        if phase_num < start_phase:
                            continue
                        if state.stop_training_flag:
                            break

                        phase_cfg = PHASES[phase_num]
                        state.current_phase = phase_num
                        state.phase_name = PHASES[phase_num].name
                        state.phase_reward_history.clear()

                        _append_log({'type': 'info', 'message': f'=== Phase {phase_num}: {PHASES[phase_num].name} ==='})
                        thread_safe_broadcast({
                            'type': 'phase_change',
                            'phase': phase_num,
                            'name': PHASES[phase_num].name,
                            'episode': state.training_stats.get('total_episodes', 0)
                        })

                        # Create phase-appropriate environment
                        if phase_num == 1:
                            env = PhaseOneEnv(state.config, repo_manager)
                        elif phase_num == 2:
                            env = PhaseTwoEnv(state.config, repo_manager)
                        elif phase_num == 3:
                            env = PhaseThreeEnv(state.config, repo_manager)
                        else:
                            env = CodeEnv(state.config, repo_manager)

                        # Override config for this phase (restored in finally below)
                        orig_max_turns = state.config.environment.max_turns
                        orig_group_size = state.config.training.grpo.group_size

                        try:
                            state.config.environment.max_turns = phase_cfg.max_turns
                            state.config.training.grpo.group_size = phase_cfg.group_size

                            # Get tasks for this phase
                            if phase_cfg.pr_ids == ["all"]:
                                tasks = loader.load_in_curriculum_order()
                            else:
                                tasks = [loader.load_pr(pr_id) for pr_id in phase_cfg.pr_ids]

                            curriculum = CurriculumManager(state.config, loader)

                            trainer = GRPOTrainer(
                                config=state.config,
                                policy=policy,
                                env=env,
                                pr_tasks=tasks
                            )
                            state.trainer = trainer
                            trainer.set_websocket_callback(thread_safe_broadcast)

                            _append_log({'type': 'info', 'message': f'Phase {phase_num}: Training started (max_turns={phase_cfg.max_turns}, group_size={phase_cfg.group_size})'})

                            # Train with advancement checking
                            max_steps = 500
                            step = 0
                            advanced = False

                            while step < max_steps and not state.stop_training_flag:
                                task = tasks[0] if tasks else None
                                if task is None:
                                    break

                                rollouts = trainer._collect_rollouts(task)
                                if not rollouts:
                                    continue

                                trainer._compute_advantages(rollouts)
                                update_stats = trainer._grpo_update(rollouts)
                                trainer._log_progress(rollouts, update_stats)
                                step += 1

                                # Check advancement
                                rewards_in_window = list(state.phase_reward_history)
                                if len(rewards_in_window) >= phase_cfg.advancement_required:
                                    met = sum(1 for r in rewards_in_window if r >= phase_cfg.advancement_threshold)
                                    if met >= phase_cfg.advancement_required:
                                        _append_log({'type': 'info', 'message': f'Phase {phase_num} COMPLETE! ({met} of {len(rewards_in_window)} above {phase_cfg.advancement_threshold})'})
                                        advanced = True
                                        break

                            # Save phase checkpoint
                            checkpoint_dir = state.config.checkpoints_path / f"phase_{phase_num}"
                            checkpoint_dir.mkdir(parents=True, exist_ok=True)
                            policy.save(str(checkpoint_dir / "model"))
                            _append_log({'type': 'info', 'message': f'Phase {phase_num} checkpoint saved'})

                            if not advanced and not state.stop_training_flag:
                                _append_log({'type': 'info', 'message': f'Phase {phase_num}: Max steps reached without advancement, continuing to next phase'})
                        finally:
                            # Restore config
                            state.config.environment.max_turns = orig_max_turns
                            state.config.training.grpo.group_size = orig_group_size

                            env.cleanup()

                    _append_log({'type': 'training_complete', 'result': {'mode': 'phased', 'final_phase': state.current_phase}})

                else:
                    # ---- REGULAR TRAINING ----
                    curriculum = CurriculumManager(state.config, loader)
                    env = CodeEnv(state.config, repo_manager)
                    _append_log({'type': 'info', 'message': 'Environment ready'})

                    _append_log({'type': 'info', 'message': 'Using GRPO algorithm'})
                    trainer = GRPOTrainer(
                        config=state.config,
                        policy=policy,
                        env=env,
                        pr_tasks=curriculum.get_remaining_tasks()
                    )
                    state.trainer = trainer
                    trainer.set_websocket_callback(thread_safe_broadcast)

                    _append_log({'type': 'info', 'message': 'Starting training loop...'})
                    result = trainer.train()

                    with state.lock:
                        state.training_stats = result.get('stats', {})
                    _append_log({'type': 'training_complete', 'result': result})

            except Exception as e:
                logger.error(f"Training error: {e}")
                logger.error(traceback.format_exc())
                _append_log({
                    'type': 'training_error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            finally:
                # Single place where is_training is cleared (stop endpoint only sets the flag)
                with state.lock:
                    state.is_training = False
                    state.trainer = None
                    state.current_phase = 0
                    state.phase_name = ""
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

        # Only signal: the training thread's finally block clears is_training
        with state.lock:
            state.stop_training_flag = True

            # Signal the trainer to stop gracefully
            if state.trainer is not None:
                state.trainer.stop_requested = True

        return {"status": "stopping", "message": "Stop signal sent; training will halt shortly"}

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

        with state.lock:
            logs_snapshot = list(state.recent_logs)

        for log in logs_snapshot:
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
        """List available phase training checkpoints for resume support."""
        if not state.config:
            raise HTTPException(status_code=500, detail="Config not loaded")

        checkpoints_dir = state.config.checkpoints_path
        completed_phases: List[int] = []
        latest_checkpoint: Optional[str] = None

        if checkpoints_dir.exists():
            # Scan for phase_N/model directories
            for entry in sorted(checkpoints_dir.iterdir()):
                if entry.is_dir() and entry.name.startswith("phase_"):
                    try:
                        phase_num = int(entry.name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    model_dir = entry / "model"
                    if model_dir.exists() and model_dir.is_dir():
                        # Verify it has adapter files (at minimum adapter_config.json)
                        has_adapter = (model_dir / "adapter_config.json").exists()
                        has_model = (model_dir / "adapter_model.safetensors").exists() or (model_dir / "adapter_model.bin").exists()
                        if has_adapter or has_model:
                            completed_phases.append(phase_num)

        completed_phases.sort()
        has_checkpoints = len(completed_phases) > 0
        resume_phase = (max(completed_phases) + 1) if completed_phases else 1
        if completed_phases:
            latest_checkpoint = str(checkpoints_dir / f"phase_{max(completed_phases)}" / "model")

        return {
            "has_checkpoints": has_checkpoints,
            "completed_phases": completed_phases,
            "resume_phase": resume_phase,
            "latest_checkpoint": latest_checkpoint,
        }

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
    """Append a log entry with a timestamp and sequence number to the recent_logs deque."""
    entry.setdefault('timestamp', datetime.now().isoformat())
    with state.lock:
        state.log_seq += 1
        entry['_seq'] = state.log_seq
        state.recent_logs.append(entry)


def run_server(host: str = None, port: int = None, config_path: str = None):
    """Run the API server.

    Args:
        host: Host to bind to (defaults to config.logging.websocket.host)
        port: Port to bind to (defaults to config.logging.websocket.port)
        config_path: Path to config file
    """
    config = load_config(config_path) if config_path else load_config()

    # Fall back to websocket config values if not explicitly provided
    if host is None:
        host = config.logging.websocket.host
    if port is None:
        port = config.logging.websocket.port

    app = create_app(config)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
