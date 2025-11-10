#!/usr/bin/env python3
"""
W&B Sweep Launcher for ST-GCN Spindle Detection
-----------------------------------------------
This script connects your W&B sweep configuration (from config.yaml)
with the training function in trainer.py.
"""

import wandb
import subprocess
import yaml
import os

CONFIG_PATH = "config.yaml"  # path to your base config


def load_sweep_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    sweep_cfg = cfg.get("sweep", {})
    sweep_cfg["project"] = cfg["logging"]["wandb"]["project"]
    sweep_cfg["entity"] = cfg["logging"]["wandb"]["entity"]
    return sweep_cfg


def train():
    """Single training run launched by W&B agent."""
    # Just run your existing trainer.py with the config
    subprocess.run(["python", "trainer.py", "--config", CONFIG_PATH])


if __name__ == "__main__":
    sweep_cfg = load_sweep_config()

    # Create sweep on W&B
    sweep_id = wandb.sweep(sweep_cfg, project=sweep_cfg["project"], entity=sweep_cfg["entity"])
    print(f"Created sweep: {sweep_id}")

    # Start agent (launch multiple runs)
    wandb.agent(sweep_id, function=train, count=sweep_cfg.get("count", 10))
