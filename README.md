# Code-RL-Ground

An RL-based code learning environment where a small language model (Qwen2.5-3B) learns to master a repository by solving PRs sequentially using **GRPO (Group Relative Policy Optimization)**.

## Overview

This project trains a model to become a "master of the repository" by:
1. Learning to solve PRs in dependency order
2. Using tools (read/write/edit files, run code, search)
3. Running multi-turn episodes with feedback
4. Using GRPO for efficient policy updates with LoRA adapters
5. Passing a championship round after solving all PRs

## Architecture

```
code-rl-ground/
├── configs/           # Configuration files
├── dataset/           # Training data
│   ├── base_repo/     # The repository to learn
│   ├── prs/           # PR task definitions
│   └── index.json     # PR metadata & dependencies
├── src/               # Python source
│   ├── agent/         # LLM policy & GRPO trainer
│   ├── data/          # PR loader, curriculum, augmentation
│   ├── environment/   # Gym-like env, sandbox, tools
│   ├── rewards/       # Reward functions
│   ├── server/        # FastAPI + WebSocket
│   └── utils/         # Config, logging, repo state
├── ui/                # React + Vite dashboard
└── scripts/           # Training scripts
```

## Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# UI dependencies
cd ui && npm install
```

### 2. Configure

Edit `configs/config.yaml` to set:
- Model name and device (MPS for M4 Max)
- Training hyperparameters
- Curriculum strategy

### 3. Run Training

**Option A: Command line**
```bash
python scripts/train.py --config configs/config.yaml
```

**Option B: With dashboard**
```bash
# Start API server (serves both API and UI)
python -m src.server.api

# Or for UI development:
cd ui && npm run dev
```

Then open http://localhost:8000 and click "Start Training"

### 4. Evaluate

```bash
python scripts/evaluate.py --model outputs/final_model --output results.json
```

## Configuration

Key settings in `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-3B"
  device: "mps"
  quantization: "4bit"  # or null for MPS

training:
  algorithm: "grpo"     # Group Relative Policy Optimization
  learning_rate: 1.0e-5
  
  grpo:
    group_size: 4       # K completions per prompt
    beta: 0.1           # KL penalty coefficient
    clip_range: 0.2     # PPO-style clipping
  
  lora:
    enabled: true
    r: 32
    alpha: 64
    
environment:
  mode: "multi_turn"    # Multi-turn episodes
  max_turns: 10         # Max turns per episode
  
curriculum:
  strategy: "dependency"       # Follow PR dependencies
  strict_progression: true     # Must solve before advancing
  solve_threshold: 0.9         # Reward needed to "solve"
  min_consecutive_solves: 3    # Confirmations needed
```

## How GRPO Works

### Algorithm Overview

GRPO (Group Relative Policy Optimization) from DeepSeek-R1:

1. **Generate K completions** for the same prompt
2. **Run multi-turn episodes** (up to 10 turns each)
3. **Compute rewards** for each completion
4. **Normalize advantages** within the group: `A_i = (r_i - mean) / std`
5. **Update LoRA adapters** using clipped surrogate objective

```
For each training step:
    1. Generate K=4 episode rollouts
    2. Each episode: model uses tools until submit() or max_turns
    3. Compute rewards from syntax, tests, file matching
    4. Group-relative advantage: A = (r - mean(r)) / std(r)
    5. Token-level loss: -min(ratio * A, clip(ratio) * A)
    6. Single optimizer.step() updates LoRA weights
```

### Update Frequency

- **1 step** = 4 episodes (group_size) × 1-10 turns each
- **LoRA update** happens once per step
- Only LoRA adapters are trained (base model frozen)

### Training Loop

1. **Curriculum** selects next PR to solve
2. **Environment** resets to repo state after previous PRs
3. **Model** generates actions using tools (read_file, edit_file, submit)
4. **Environment** executes actions, returns observations
5. **Reward** computed when model calls submit()
6. **GRPO** updates LoRA policy based on group-relative advantages

### Strict Progression

The model must achieve:
- Reward ≥ 0.9 (solve threshold)
- 3 consecutive successful solves
- All dependent PRs already solved

Only then does it advance to the next PR.

### Championship Round

After solving all PRs:
1. All PRs shuffled randomly
2. Model must re-solve each in single turn
3. Tests model's true "mastery" of the repo

## Tool Format

The model uses tools with this format:
```
<tool>tool_name(param="value")</tool>
```

Available tools:
- `read_file(path)` - Read a file
- `write_file(path, content)` - Write a new file
- `edit_file(path, old_content, new_content)` - Edit existing file
- `run_python(code)` - Run Python code
- `submit()` - Submit solution for reward

## Dataset Format

Each PR in `dataset/prs/pr_XXX.json`:

```json
{
  "pr_id": "PR-001",
  "title": "Add reverse_string function",
  "description": "Implement string reversal",
  "difficulty": 1,
  "depends_on": [],
  "files_changed": ["pyutils/strings.py"],
  "expected_changes": { ... },
  "test_cases": [ ... ]
}
```

## UI Features

- **PR Progress**: Visual tracker of solved PRs
- **Live Generation**: Real-time model output streaming
- **Reward Charts**: Training curves (loss, rewards, solve rate)
- **Tool Logs**: See every tool call and result
- **Metrics Panel**: Steps, episodes, average reward
- **Controls**: Start/stop training from browser

## License

MIT
