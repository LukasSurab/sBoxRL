# sBoxRL
This repository implements a modular pipeline for evolving 8×8 S-boxes using reinforcement learning. It supports multiple RL algorithms (DQN, A2C, PPO) via Stable-Baselines3 and provides customizable reward modes, checkpointing, and statistical reporting.

## Overview

- **Modular Design**: Split into four main modules:
  - `sbox_utils.py`: S-box generation, spectral & differential metrics.
  - `sbox_env.py`: Custom Gym environment (`SBoxEnv`).
  - `callbacks.py`: `StatsCallback` for checkpoints, JSON/CSV summaries & plots.
  - `train_*.py`: Training scripts for DQN (`train_dqn.py`), A2C (`train_a2c.py`), PPO (`train_ppo.py`).
- **Algorithms**: Easily switch between DQN, A2C, PPO by choosing the appropriate `train_*.py`.
- **Customisation**: Tweak architecture, hyperparameters, reward settings, allowed gates, and more.

---

##  Prerequisites

- **Python 3.7+**
- **pip** (with `--upgrade`) or Conda
- Virtual environment tool (e.g., `venv`, `virtualenv`, or Conda)

---

##  Dependency Installation
python -m pip install --upgrade pip
pip install -r requirements.txt

List of packages:
gymnasium
numpy
torch
sympy
matplotlib
stable-baselines3

#  Configuration Guide

This section describes configuring the training scripts for evolving 8×8 S-boxes via RL.

---

## 1. Training Scripts (`train_*.py`)

Each script corresponds to one algorithm:  
- **DQN**: `train_dqn.py`  
- **A2C**: `train_a2c.py`  
- **PPO**: `train_ppo.py`  

## Possible customisations:
You can customise every aspect of training before running.

Edit the top of your chosen `train_*.py` to adjust the hyperparameters. They are located in
`*_hyperparams{}`.

Define your neural network via `policy_kwargs` like such:
```
policy_kwargs = dict(
    activation_fn=nn.LeakyReLU,
    net_arch=[512, 256, 128]   # hidden layer sizes
)
```
When you instantiate `SBoxEnv`, you can pass:
```
env = SBoxEnv(
    max_steps=100000,                   # max operations per episode
    reward_mode="spectral_combined",    # e.g., "nonlinearity", "du", "spectral_linear_du", etc.
    init_random_ops=True,               # start with random gate sequence
    random_ops_count=0,                 # how many random ops on reset
    non_improvement_limit=50,           # early termination threshold
    reward_config={                    # custom reward weights & thresholds
        'w_nonl': 2.0,
        'w_du': 1.5,
        'w_gate': 0.15,
        'w_explore': 0.8,
        'w_no_improve': 0.005,
        # ... other bonus/threshold settings ...
    },
    allowed_gates=["XOR", "TOFFOLI", "FREDKIN"],  # gates the agent may use
)
```

Checkpoint interval: how often (in timesteps) to save the model & stats.

Total timesteps: overall training budget.
