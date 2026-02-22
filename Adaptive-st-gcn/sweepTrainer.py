#!/usr/bin/env python3
"""
sweepTrainer_timepoint.py - Fixed version

Bugs fixed:
1. Missing error handling for missing sweep ID
2. No validation that sweep_id is not empty
3. Missing cleanup of temp files on error
4. No handling for wandb.init() failures
5. Better error messages
"""
import os
import argparse
import tempfile
import wandb
import subprocess
import yaml
import sys

BASE_CONFIG = "config.yaml"

def apply_sweep_to_config(base_config_path: str, sweep_params: dict) -> str:
    """
    Apply sweep parameters to config and create temporary config file.

    Args:
        base_config_path: Path to base config.yaml
        sweep_params: Dictionary of sweep parameters from W&B

    Returns:
        Path to temporary config file
    """
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("train", {})
    cfg.setdefault("model", {})
    cfg.setdefault("path", {})

    # Apply sweep parameters
    for k, v in sweep_params.items():
        if k in cfg["train"]:
            cfg["train"][k] = v
        elif k in cfg["model"]:
            cfg["model"][k] = v
        elif k in cfg["path"]:
            cfg["path"][k] = v
        else:
            cfg["train"][k] = v

    # Unique save folder per run
    if wandb.run is not None:
        run_id = wandb.run.id
        base_save = cfg["path"].get("save", "./result/")
        cfg["path"]["save"] = os.path.join(base_save, run_id)

    # Create temporary config file
    fd, tmp_path = tempfile.mkstemp(prefix="SS3_sweep_", suffix=".yaml")
    os.close(fd)

    try:
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception as e:
        os.remove(tmp_path)
        raise RuntimeError(f"Failed to write temporary config: {e}")

    return tmp_path


def train_agent():
    """
    Train agent function called by W&B sweep.
    """
    tmp_cfg = None

    try:
        # Initialize W&B run
        run = wandb.init()

        if run is None:
            raise RuntimeError("Failed to initialize W&B run")

        print(f"\n{'=' * 60}")
        print(f"Starting sweep run: {run.id}")
        print(f"Sweep config: {dict(wandb.config)}")
        print(f"{'=' * 60}\n")

        # Create temporary config with sweep parameters
        #tmp_cfg = apply_sweep_to_config("config.yaml", dict(wandb.config))
        tmp_cfg = apply_sweep_to_config(BASE_CONFIG, dict(wandb.config))

        print(f"Created temporary config: {tmp_cfg}")

        # Run training
        import runpy, sys

        # run train.py inside the same process so it shares wandb.run
        sys.argv = ["train.py", "-c", tmp_cfg, "-g", "0"]
        runpy.run_path("train.py", run_name="__main__")


    except Exception as e:
        print(f"\n❌ Error in train_agent: {e}")
        raise

    finally:
        # Cleanup temporary config
        if tmp_cfg and os.path.exists(tmp_cfg):
            try:
                os.remove(tmp_cfg)
                print(f"Cleaned up temporary config: {tmp_cfg}")
            except Exception as e:
                print(f"Warning: Failed to remove temp config {tmp_cfg}: {e}")

        # Finish W&B run
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Error finishing W&B run: {e}")


def create_sweep(sweep_cfg, entity, project):
    """
    Create a new W&B sweep.

    Args:
        sweep_cfg: Sweep configuration dictionary
        entity: W&B entity name
        project: W&B project name

    Returns:
        sweep_id: ID of created sweep
    """
    try:
        sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)
        print("\n" + "=" * 60)
        print("✅ Sweep created successfully!")
        print(f"Sweep ID: {sweep_id}")
        print(f"Entity: {entity}")
        print(f"Project: {project}")
        print("=" * 60)

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
    """
    Run an existing W&B sweep.

    Args:
        sweep_id: ID of the sweep to run
        entity: W&B entity name
        project: W&B project name
    """
    if not sweep_id or sweep_id.strip() == "":
        print("\n❌ ERROR: No sweep ID provided!")
        print("\nYou need to:")
        print("1. First create a sweep: python sweepTrainer_timepoint.py --create")
        print("2. Copy the sweep ID")
        print("3. Paste it in config.yaml under sweep_wandb -> id")
        print("4. Then run: python sweepTrainer_timepoint.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Starting W&B sweep agent")
    print(f"Sweep ID: {sweep_id}")
    print(f"Entity: {entity}")
    print(f"Project: {project}")
    print("=" * 60 + "\n")

    try:
        wandb.agent(
            sweep_id,
            function=train_agent,
            entity=entity,
            project=project
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Sweep interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Sweep failed: {e}")
        print("\nTroubleshooting:")
        print("1. Verify sweep ID is correct")
        print("2. Check W&B connection: wandb login")
        print("3. Ensure config.yaml is valid")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="W&B Sweep Trainer for GraphSleepNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new sweep
  python sweepTrainer_timepoint.py --create

  # Run an existing sweep (requires sweep ID in config.yaml)
  python sweepTrainer_timepoint.py

  # Run with custom config
  python sweepTrainer_timepoint.py --config my_config.yaml
        """
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new sweep (outputs sweep ID)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        sys.exit(1)

    # Validate config structure
    if "sweep_wandb" not in cfg:
        print("❌ Error: 'sweep_wandb' section missing in config.yaml")
        print("\nAdd this to your config.yaml:")
        print("""
sweep_wandb:
  id: ""  # Leave empty initially, fill after creating sweep
  method: bayes
  metric:
    name: mean_test_pr_auc
    goal: maximize
  parameters:
    learn_rate:
      distribution: log_uniform_values
      min: 1e-5
      max: 0.003
        """)
        sys.exit(1)

    sweep_cfg = cfg["sweep_wandb"]

    # W&B configuration
    entity = "subahininadarajh-basel-university"
    project = "Graph_learning_graph sweep_fixed_lr"

    # Execute based on mode
    if args.create:
        # Create new sweep
        print("\n🔄 Creating new W&B sweep...")
        create_sweep(sweep_cfg, entity, project)
    else:
        # Run existing sweep
        sweep_id = sweep_cfg.get("id", "").strip()
        run_sweep(sweep_id, entity, project)

    global BASE_CONFIG
    BASE_CONFIG = args.config

if __name__ == "__main__":
    main()