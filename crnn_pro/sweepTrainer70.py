
#!/usr/bin/env python3
"""
W&B Sweep-Compatible Training Script for CRNN (crrn_70.py + config70.yaml)
- Loads base config70.yaml
- Applies W&B sweep overrides
- Runs crrn_70.train_and_eval(cfg)
"""

import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

import argparse
import copy
import os
import sys
import yaml
import wandb

# IMPORTANT: import your new training module
import crnn_70 as Crnn
from crnn_70 import safe_cfg


def apply_sweep_config(base_cfg, sweep_params):
    """Apply W&B sweep parameters to base config."""
    cfg = copy.deepcopy(base_cfg)

    for key, value in sweep_params.items():
        # Skip W&B internal params
        if key.startswith("_"):
            continue

        # Handle nested keys: "trainer.lr" -> cfg["trainer"]["lr"]
        keys = key.split(".")
        current = cfg
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]

        final_key = keys[-1]

        # Special handling for pos_weight = "auto"
        if final_key == "pos_weight" and value == "auto":
            current[final_key] = "auto"
        else:
            current[final_key] = value

    return cfg


def train():
    """Main training function called by W&B sweep agent."""
    run = wandb.init()  # init ONCE here

    # Load base config (from env or default)
    config_path = os.getenv("CRNN_CONFIG_PATH", "config70.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f) or {}

    base_cfg = safe_cfg(base_cfg)

    sweep_params = dict(wandb.config)

    print("\n" + "=" * 60)
    print(f"[SWEEP] Run: {run.name} (ID: {run.id})")
    print("=" * 60)
    print("Sweep parameters:")
    for k, v in sorted(sweep_params.items()):
        if not k.startswith("_"):
            print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    # Apply sweep overrides
    cfg = apply_sweep_config(base_cfg, sweep_params)

    # Prevent double wandb.init inside crrn_70.py
    wb_cfg = cfg.setdefault("logging", {}).setdefault("wandb", {})
    wb_cfg["enabled"] = False

    # Make output dir unique per run (avoid overwrite)
    cfg.setdefault("paths", {})
    cfg["paths"]["out_dir"] = os.path.join(
        cfg["paths"].get("out_dir", "./checkpoints"),
        f"sweep_{run.id}"
    )

    try:
        print("[SWEEP] Starting training...")
        Crnn.train_and_eval(cfg)
        print(f"[SWEEP] Training completed successfully for run {run.name}")
    except Exception as e:
        print(f"[SWEEP] ERROR in run {run.name}: {e}")
        import traceback
        traceback.print_exc()
        try:
            wandb.log({"error": str(e), "error_type": type(e).__name__})
        except Exception:
            pass
        raise
    finally:
        wandb.finish()


def init_sweep(config_path="config70.yaml", project=None, entity=None):
    """Initialize a W&B sweep from config70.yaml"""
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    sweep_cfg = cfg.get("sweep", {})
    if not sweep_cfg:
        print("ERROR: No 'sweep' section found in config70.yaml")
        sys.exit(1)

    if not sweep_cfg.get("enabled", False):
        print("WARNING: sweep.enabled is false in config70.yaml; forcing enabled=true")
        sweep_cfg["enabled"] = True

    sweep_config = {
        "method": sweep_cfg.get("method", "bayes"),
        "metric": sweep_cfg.get("metric", {"name": "val/f1", "goal": "maximize"}),
        "parameters": sweep_cfg.get("parameters", {}),
        "program": "sweepTrainer_timepoint.py",
    }

    wb_cfg = cfg.get("logging", {}).get("wandb", {})
    project = project or wb_cfg.get("project", "CRNN-sweep")
    entity = entity or wb_cfg.get("entity") or cfg.get("project", {}).get("entity")

    if not entity:
        print("ERROR: No W&B entity specified in config (logging.wandb.entity or project.entity)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Initializing W&B Sweep")
    print("=" * 60)
    print(f"Project: {project}")
    print(f"Entity:  {entity}")
    print(f"Method:  {sweep_config['method']}")
    print(f"Metric:  {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print("\nParameters to sweep:")
    for param, conf in sweep_config["parameters"].items():
        if isinstance(conf, dict) and "values" in conf:
            print(f"  {param}: {conf['values']}")
        elif isinstance(conf, dict) and "min" in conf and "max" in conf:
            print(f"  {param}: [{conf['min']}, {conf['max']}]")
        else:
            print(f"  {param}: {conf}")
    print("=" * 60 + "\n")

    try:
        sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    except Exception as e:
        print(f"ERROR creating sweep: {e}")
        print("Make sure you're logged in: wandb login")
        sys.exit(1)

    print("✓ Sweep created successfully!")
    print(f"Sweep ID: {sweep_id}")
    print(f"To start agents:\n  wandb agent {entity}/{project}/{sweep_id}")
    return sweep_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Initialize a new sweep")
    parser.add_argument("--config", type=str, default="config70.yaml")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)

    # Accept sweep params (ignored here; wandb.config is used)
    parser.add_argument("--loss.focal.alpha", type=float, default=None)
    parser.add_argument("--loss.focal.gamma", type=float, default=None)
    parser.add_argument("--loss.name", type=str, default=None)
    parser.add_argument("--loss.weighted_bce.adaptive", type=str, default=None)
    parser.add_argument("--loss.weighted_bce.pos_weight", type=str, default=None)
    parser.add_argument("--model.fpn_ch", type=int, default=None)
    parser.add_argument("--model.rnn_hidden", type=int, default=None)
    parser.add_argument("--model.rnn_layers", type=int, default=None)
    parser.add_argument("--model.use_se", type=str, default=None)
    parser.add_argument("--trainer.batch_size", type=int, default=None)
    parser.add_argument("--trainer.lr", type=float, default=None)
    parser.add_argument("--trainer.sampler", type=str, default=None)
    parser.add_argument("--trainer.weight_decay", type=float, default=None)

    args, _unknown = parser.parse_known_args()

    # Pass config path to train() via env var (used when wandb agent runs)
    os.environ["CRNN_CONFIG_PATH"] = args.config

    if args.init:
        init_sweep(args.config, args.project, args.entity)
    else:
        train()


if __name__ == "__main__":
    main()
