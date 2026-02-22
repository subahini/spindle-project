#!/usr/bin/env python3
"""
sweepTrainer_timepoint.py

Same as sweepTrainer_timepoint.py but runs train_timepoint.py (time-point pipeline).
- Applies W&B sweep params into a temporary config.yaml
- Runs train_timepoint.py in-process (so it shares wandb.run)
"""

import os
import argparse
import tempfile
import wandb
import yaml
import sys
import runpy

BASE_CONFIG = "config.yaml"


def apply_sweep_to_config(base_config_path: str, sweep_params: dict) -> str:
    """Apply sweep parameters to config and create a temporary yaml file."""
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("train", {})
    cfg.setdefault("model", {})
    cfg.setdefault("path", {})

    # Apply sweep parameters into train/model/path if key exists, else default to train
    for k, v in sweep_params.items():
        if k in cfg["train"]:
            cfg["train"][k] = v
        elif k in cfg["model"]:
            cfg["model"][k] = v
        elif k in cfg["path"]:
            cfg["path"][k] = v
        else:
            cfg["train"][k] = v

    # Unique save folder per run (so sweep runs don't overwrite)
    if wandb.run is not None:
        run_id = wandb.run.id
        base_save = cfg["path"].get("save", "./result/")
        cfg["path"]["save"] = os.path.join(base_save, run_id)

    fd, tmp_path = tempfile.mkstemp(prefix="SS3_timepoint_sweep_", suffix=".yaml")
    os.close(fd)

    try:
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to write temporary config: {e}")

    return tmp_path


def train_agent():
    """Function called by wandb.agent()."""
    tmp_cfg = None
    try:
        run = wandb.init()
        if run is None:
            raise RuntimeError("Failed to initialize W&B run")

        print("\n" + "=" * 60)
        print(f"Starting TIMEPOINT sweep run: {run.id}")
        print(f"Sweep config: {dict(wandb.config)}")
        print("=" * 60 + "\n")

        tmp_cfg = apply_sweep_to_config(BASE_CONFIG, dict(wandb.config))
        print(f"Created temporary config: {tmp_cfg}")

        # Run train_timepoint.py inside same process so it shares wandb.run
        sys.argv = ["train_timepoint.py", "-c", tmp_cfg, "-g", "0"]
        runpy.run_path("train_timepoint.py", run_name="__main__")

    except Exception as e:
        print(f"\n❌ Error in train_agent: {e}")
        raise

    finally:
        if tmp_cfg and os.path.exists(tmp_cfg):
            try:
                os.remove(tmp_cfg)
                print(f"Cleaned up temporary config: {tmp_cfg}")
            except Exception as e:
                print(f"Warning: Failed to remove temp config {tmp_cfg}: {e}")

        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Error finishing W&B run: {e}")


def create_sweep(sweep_cfg, entity, project):
    """Create a new sweep and print instructions."""
    try:
        sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)
        print("\n" + "=" * 60)
        print("✅ Timepoint sweep created successfully!")
        print(f"Sweep ID: {sweep_id}")
        print(f"Entity: {entity}")
        print(f"Project: {project}")
        print("=" * 60)
        print("\nNext steps:")
        print(f"1. Copy the sweep ID: {sweep_id}")
        print(f"2. Paste it in config.yaml under sweep_wandb -> id")
        print(f"3. Run: python sweepTrainer_timepoint.py")
        print("=" * 60 + "\n")
        return sweep_id
    except Exception as e:
        print(f"\n❌ Failed to create sweep: {e}")
        print("\nTroubleshooting:")
        print("1. Check your W&B login: wandb login")
        print("2. Verify entity and project names")
        print("3. Check internet connection")
        sys.exit(1)


def run_sweep(sweep_id, entity, project):
    """Run an existing sweep."""
    if not sweep_id or sweep_id.strip() == "":
        print("\n❌ ERROR: No sweep ID provided!")
        print("\nYou need to:")
        print("1. First create a sweep: python sweepTrainer_timepoint.py --create")
        print("2. Copy the sweep ID")
        print("3. Paste it in config.yaml under sweep_wandb -> id")
        print("4. Then run: python sweepTrainer_timepoint.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Starting W&B TIMEPOINT sweep agent")
    print(f"Sweep ID: {sweep_id}")
    print(f"Entity: {entity}")
    print(f"Project: {project}")
    print("=" * 60 + "\n")

    try:
        wandb.agent(
            sweep_id,
            function=train_agent,
            entity=entity,
            project=project,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Sweep interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Sweep failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="W&B Sweep Trainer for TIMEPOINT GraphSleepNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new sweep
  python sweepTrainer_timepoint.py --create

  # Run an existing sweep (requires sweep ID in config.yaml)
  python sweepTrainer_timepoint.py

  # Run with custom config
  python sweepTrainer_timepoint.py --config config_timepoint.yaml
"""
    )
    parser.add_argument("--create", action="store_true", help="Create a new sweep (outputs sweep ID)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    args = parser.parse_args()

    global BASE_CONFIG
    BASE_CONFIG = args.config

    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        sys.exit(1)

    if "sweep_wandb" not in cfg:
        print("❌ Error: 'sweep_wandb' section missing in config.yaml")
        sys.exit(1)

    sweep_cfg = cfg["sweep_wandb"]

    # Hardcoded like your current sweepTrainer_timepoint.py (change if you want)
    entity = "subahininadarajh-basel-university"
    project = cfg.get("train", {}).get("project", "spindle-timepoint")

    if args.create:
        print("\n🔄 Creating new W&B TIMEPOINT sweep...")
        create_sweep(sweep_cfg, entity, project)
    else:
        sweep_id = sweep_cfg.get("id", "").strip()
        run_sweep(sweep_id, entity, project)


if __name__ == "__main__":
    main()
