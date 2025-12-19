"""Configuration loading and management."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-3B"
    backend: str = "transformers"
    quantization: str = "4bit"
    device: str = "mps"
    max_context_length: int = 8192
    generation: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True
    })


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass
class GRPOConfig:
    group_size: int = 4
    beta: float = 0.1
    clip_range: float = 0.2


@dataclass
class CheckpointConfig:
    enabled: bool = True
    save_every_n_steps: int = 100
    keep_last_n: int = 3
    save_optimizer: bool = True
    save_scheduler: bool = True


@dataclass
class TrainingConfig:
    algorithm: str = "grpo"
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs_per_pr: int = 10
    max_episodes: int = 1000
    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class SandboxConfig:
    enabled: bool = True
    backend: str = "subprocess"
    timeout_per_execution: int = 30
    max_memory_mb: int = 512
    allowed_imports: List[str] = field(default_factory=lambda: [
        "os", "sys", "re", "json", "math", "collections", 
        "itertools", "functools", "pathlib", "typing", "time"
    ])


@dataclass
class MultiTurnFeedbackConfig:
    show_correct_incorrect: bool = True
    show_error_messages: bool = True
    show_partial_reward: bool = True
    show_test_results: bool = True
    show_diff_hints: bool = False


@dataclass
class ToolsConfig:
    enabled: bool = True
    available: List[str] = field(default_factory=lambda: [
        "read_file", "write_file", "edit_file", 
        "list_directory", "run_python", "search_code", "submit"
    ])


@dataclass
class EnvironmentConfig:
    mode: str = "single_turn"
    max_turns: int = 5
    timeout_seconds: int = 60
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    multi_turn_feedback: MultiTurnFeedbackConfig = field(default_factory=MultiTurnFeedbackConfig)


@dataclass
class RewardWeights:
    syntax_valid: float = 0.1
    compiles: float = 0.2
    tests_pass: float = 0.3
    files_match: float = 0.2
    exact_match: float = 0.2


@dataclass
class PartialCreditConfig:
    enabled: bool = True
    line_match_weight: float = 0.5
    function_match_weight: float = 0.5


@dataclass
class RewardsConfig:
    weights: RewardWeights = field(default_factory=RewardWeights)
    partial_credit: PartialCreditConfig = field(default_factory=PartialCreditConfig)


@dataclass
class CurriculumConfig:
    enabled: bool = True
    strategy: str = "dependency"
    respect_dependencies: bool = True
    strict_progression: bool = True
    solve_threshold: float = 0.9
    min_consecutive_solves: int = 3
    max_attempts_per_pr: int = 1000


@dataclass
class AugmentationConfig:
    enabled: bool = True
    strategies: List[str] = field(default_factory=lambda: [
        "variable_renaming", "docstring_variation", "whitespace_variation"
    ])
    multiplier: int = 3
    seed: int = 42


@dataclass
class MasteryTestConfig:
    num_attempts_per_pr: int = 5
    require_all_prs: bool = True
    success_threshold_per_pr: float = 0.9
    overall_success_threshold: float = 1.0


@dataclass
class MasteryConfig:
    enabled: bool = True
    eval_frequency: int = 500
    full_eval_at_end: bool = True
    test_config: MasteryTestConfig = field(default_factory=MasteryTestConfig)


@dataclass
class ChampionshipConfig:
    enabled: bool = True
    trigger: str = "all_prs_solved"
    mode: str = "sequential"
    allow_retries: bool = False
    save_transcript: bool = True


@dataclass
class ModelSavingConfig:
    save_final: bool = True
    final_model_path: str = "./checkpoints/final_model"
    save_on_mastery: bool = True
    save_format: str = "safetensors"


@dataclass
class WebSocketConfig:
    enabled: bool = True
    host: str = "localhost"
    port: int = 8765


@dataclass
class LoggingConfig:
    level: str = "INFO"
    save_generations: bool = True
    save_every_n_steps: int = 100
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)


@dataclass
class UIConfig:
    enabled: bool = True
    host: str = "localhost"
    port: int = 3000
    api_port: int = 8000
    refresh_rate_ms: int = 500


@dataclass
class PathsConfig:
    dataset: str = "./dataset"
    cache: str = "./cache"
    checkpoints: str = "./checkpoints"
    logs: str = "./logs"
    ui_build: str = "./ui/build"


@dataclass
class ProjectConfig:
    name: str = "code-rl-ground"
    description: str = "RL-based code learning environment"
    version: str = "0.1.0"


@dataclass
class Config:
    """Main configuration container."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    mastery: MasteryConfig = field(default_factory=MasteryConfig)
    championship: ChampionshipConfig = field(default_factory=ChampionshipConfig)
    model_saving: ModelSavingConfig = field(default_factory=ModelSavingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Root path for resolving relative paths
    root_path: Path = field(default_factory=lambda: Path.cwd())
    
    def resolve_path(self, path: str) -> Path:
        """Resolve a relative path to absolute."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.root_path / p
    
    @property
    def dataset_path(self) -> Path:
        return self.resolve_path(self.paths.dataset)
    
    @property
    def cache_path(self) -> Path:
        return self.resolve_path(self.paths.cache)
    
    @property
    def checkpoints_path(self) -> Path:
        return self.resolve_path(self.paths.checkpoints)
    
    @property
    def logs_path(self) -> Path:
        return self.resolve_path(self.paths.logs)


def _dict_to_dataclass(cls, data: Dict[str, Any]):
    """Recursively convert a dictionary to a dataclass instance."""
    if data is None:
        return cls()
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # Check if it's a dataclass
            if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value
    
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for configs/config.yaml
        
    Returns:
        Config object with all settings
    """
    if config_path is None:
        # Try to find config in standard locations
        candidates = [
            Path.cwd() / "configs" / "config.yaml",
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent.parent / "configs" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break
        else:
            # Return default config
            return Config()
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Build config from YAML data
    config = Config(root_path=config_file.parent.parent)
    
    if 'project' in data:
        config.project = _dict_to_dataclass(ProjectConfig, data['project'])
    if 'paths' in data:
        config.paths = _dict_to_dataclass(PathsConfig, data['paths'])
    if 'model' in data:
        config.model = _dict_to_dataclass(ModelConfig, data['model'])
    if 'training' in data:
        config.training = _dict_to_dataclass(TrainingConfig, data['training'])
    if 'environment' in data:
        config.environment = _dict_to_dataclass(EnvironmentConfig, data['environment'])
    if 'rewards' in data:
        config.rewards = _dict_to_dataclass(RewardsConfig, data['rewards'])
    if 'curriculum' in data:
        config.curriculum = _dict_to_dataclass(CurriculumConfig, data['curriculum'])
    if 'augmentation' in data:
        config.augmentation = _dict_to_dataclass(AugmentationConfig, data['augmentation'])
    if 'mastery' in data:
        config.mastery = _dict_to_dataclass(MasteryConfig, data['mastery'])
    if 'championship' in data:
        config.championship = _dict_to_dataclass(ChampionshipConfig, data['championship'])
    if 'model_saving' in data:
        config.model_saving = _dict_to_dataclass(ModelSavingConfig, data['model_saving'])
    if 'logging' in data:
        config.logging = _dict_to_dataclass(LoggingConfig, data['logging'])
    if 'ui' in data:
        config.ui = _dict_to_dataclass(UIConfig, data['ui'])
    
    # Create necessary directories
    for dir_path in [config.cache_path, config.checkpoints_path, config.logs_path]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return config
