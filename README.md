# Code-RL-Ground

An RL-based code learning environment where a small language model (Qwen2.5-3B) learns to master a repository by solving PRs sequentially.

## Overview

This project trains a model to become a "master of the repository" by:
1. Learning to solve PRs in dependency order
2. Using tools (read/write/edit files, run code, search)
3. Receiving multi-turn feedback (correct/incorrect, error messages, partial rewards)
4. Passing a championship round after solving all PRs

## Architecture

```
code-rl-ground/
├── configs/           # Configuration files
├── dataset/           # Training data
│   ├── base_repo/     # The repository to learn
│   ├── prs/           # PR task definitions
│   └── index.json     # PR metadata & dependencies
├── src/               # Python source
│   ├── agent/         # LLM policy & PPO trainer
│   ├── data/          # PR loader, curriculum, augmentation
│   ├── environment/   # Gym-like env, sandbox, tools
│   ├── rewards/       # Reward functions
│   ├── server/        # FastAPI + WebSocket
│   └── utils/         # Config, logging, repo state
├── ui/                # React dashboard
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
# Terminal 1: Start API server
python scripts/serve.py --port 8000

# Terminal 2: Start UI (development)
cd ui && npm run dev
```

Then open http://localhost:3000 and click "Start Training"

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
  quantization: "4bit"

training:
  algorithm: "ppo"
  learning_rate: 1.0e-5
  batch_size: 2
  
curriculum:
  strategy: "dependency"       # Follow PR dependencies
  strict_progression: true     # Must solve before advancing
  solve_threshold: 0.9         # Reward needed to "solve"
  min_consecutive_solves: 3    # Confirmations needed
```

## How It Works

### Training Loop

1. **Curriculum** selects next PR to solve
2. **Environment** resets to repo state after previous PRs
3. **Model** generates actions (tool calls + reasoning)
4. **Environment** executes actions via sandbox
5. **Reward** computed from syntax, tests, diff matching
6. **PPO** updates policy based on rewards

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
- **Reward Charts**: Training curves with moving average
- **Tool Logs**: See every tool call and result
- **Controls**: Start/stop training from browser

## License

MIT
