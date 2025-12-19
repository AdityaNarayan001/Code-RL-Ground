"""FastAPI server for the training dashboard."""

import os
# Disable tokenizer parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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
        self.recent_logs: List[Dict[str, Any]] = []
        self.max_logs = 1000
        self.executor = ThreadPoolExecutor(max_workers=1)


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
                    # Broadcast new logs
                    for log in state.recent_logs[last_broadcast_idx[0]:current_len]:
                        await ws_manager.broadcast(log)
                    last_broadcast_idx[0] = current_len
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
        
        return TrainingStatus(
            is_running=state.is_training,
            current_step=state.training_stats.get('total_steps', 0),
            current_episode=state.training_stats.get('total_episodes', 0),
            current_pr=state.training_stats.get('current_pr_id'),
            solved_prs=state.training_stats.get('solved_prs', []),
            avg_reward=state.training_stats.get('avg_reward', 0.0),
            elapsed_time=elapsed
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
                best_reward=best_rewards.get(task.pr_id, 0.0),
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
        
        state.is_training = True
        state.start_time = datetime.now()
        state.training_stats = {}
        state.stop_training_flag = False
        
        # Run training in thread pool to avoid blocking
        def run_training_sync():
            from ..agent import LLMPolicy, GRPOTrainer
            from ..environment import CodeEnv
            from ..data import PRLoader, CurriculumManager
            from ..utils.repo_state import RepoStateManager
            
            loop = asyncio.new_event_loop()
            
            try:
                logger.info("Initializing training...")
                state.recent_logs.append({'type': 'info', 'message': 'Initializing training...'})
                
                # Setup components
                loader = PRLoader(state.config.dataset_path)
                repo_manager = RepoStateManager(
                    base_repo_path=state.config.dataset_path / "base_repo",
                    pr_data_path=state.config.dataset_path / "prs",
                    cache_path=state.config.cache_path
                )
                
                curriculum = CurriculumManager(state.config, loader)
                env = CodeEnv(state.config, repo_manager)
                state.recent_logs.append({'type': 'info', 'message': 'Environment ready'})
                
                logger.info("Loading model (this may take a few minutes)...")
                state.recent_logs.append({'type': 'info', 'message': 'Loading model...'})
                policy = LLMPolicy(state.config)
                policy.load()
                state.recent_logs.append({'type': 'info', 'message': 'Model loaded, setting up LoRA...'})
                policy.setup_lora()
                logger.info("Model loaded successfully!")
                state.recent_logs.append({'type': 'info', 'message': 'LoRA ready!'})
                
                # Create GRPO trainer
                state.recent_logs.append({'type': 'info', 'message': 'Using GRPO algorithm'})
                trainer = GRPOTrainer(
                    config=state.config,
                    policy=policy,
                    env=env,
                    pr_tasks=curriculum.get_remaining_tasks()
                )
                
                # Set callback for updates
                def thread_safe_broadcast(data: Dict[str, Any]):
                    state.recent_logs.append(data)
                    if len(state.recent_logs) > state.max_logs:
                        state.recent_logs.pop(0)
                    
                    # Update training stats from broadcasts
                    if data.get('type') == 'step':
                        metrics = data.get('metrics', {})
                        state.training_stats.update({
                            'total_steps': data.get('step', 0),
                            'avg_reward': metrics.get('avg_reward', metrics.get('mean_reward', 0)),
                            'solve_rate': metrics.get('solve_rate', 0)
                        })
                    elif data.get('type') == 'episode':
                        state.training_stats['total_episodes'] = data.get('episode', 0)
                        state.training_stats['current_pr_id'] = data.get('pr_id')
                        # Also update avg_reward from episodes
                        reward = data.get('reward', 0)
                        old_avg = state.training_stats.get('avg_reward', 0)
                        episodes = state.training_stats.get('total_episodes', 1)
                        # Running average
                        state.training_stats['avg_reward'] = old_avg + (reward - old_avg) / max(1, episodes)
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
                
                state.recent_logs.append({'type': 'info', 'message': 'Starting training loop...'})
                
                # Run training
                result = trainer.train()
                
                state.training_stats = result.get('stats', {})
                state.recent_logs.append({
                    'type': 'training_complete',
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                logger.error(traceback.format_exc())
                state.recent_logs.append({
                    'type': 'training_error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
            finally:
                state.is_training = False
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
        state.is_training = False
        
        return {"status": "stopped", "message": "Stop signal sent"}
    
    @app.get("/api/logs")
    async def get_logs(limit: int = 100):
        """Get recent logs."""
        return state.recent_logs[-limit:]
    
    @app.get("/api/metrics")
    async def get_metrics():
        """Get training metrics for charts."""
        # Extract metrics from logs
        rewards = []
        steps = []
        
        for log in state.recent_logs:
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
        
        return {
            'steps': steps[-100:],
            'episodes': rewards[-100:]
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
            ws_manager.disconnect(websocket)
    
    # Serve static files (React build) if available
    ui_path = state.config.resolve_path("./ui/build")
    if ui_path.exists():
        app.mount("/", StaticFiles(directory=str(ui_path), html=True), name="static")
    
    return app


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
