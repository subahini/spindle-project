#!/usr/bin/env python3
"""
W&B Sweep-Compatible Training Script for CRNN
"""
import argparse
import yaml
import wandb
import os
import sys
import copy

# Import your existing training code
import Crnn


def apply_sweep_config(base_cfg, sweep_params):
    """Apply W&B sweep parameters to base config."""
    cfg = copy.deepcopy(base_cfg)

    for key, value in sweep_params.items():
        # Skip W&B internal parameters
        if key.startswith('_'):
            continue

        # Handle nested keys (e.g., "trainer.lr")
        keys = key.split('.')
        current = cfg

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        final_key = keys[-1]

        # Handle special cases for pos_weight
        if final_key == 'pos_weight' and value == 'auto':
            current[final_key] = 'auto'
        else:
            current[final_key] = value

    return cfg


def train():
    """Main training function called by W&B sweep agent."""
    # Initialize W&B run
    run = wandb.init()

    # Load base config
    config_path = os.getenv('CRNN_CONFIG_PATH', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f) or {}

    # Make sure config is valid
    base_cfg = Crnn.safe_cfg(base_cfg)

    # Apply sweep parameters from W&B
    sweep_params = dict(wandb.config)
    print(f"\n{'=' * 60}")
    print(f"[SWEEP] Run: {run.name} (ID: {run.id})")
    print(f"{'=' * 60}")
    print("Sweep parameters:")
    for k, v in sorted(sweep_params.items()):
        if not k.startswith('_'):
            print(f"  {k}: {v}")
    print(f"{'=' * 60}\n")

    # Apply sweep parameters to config
    cfg = apply_sweep_config(base_cfg, sweep_params)

    # Update W&B config settings - DON'T call wandb.init again!
    wb_cfg = cfg.setdefault('logging', {}).setdefault('wandb', {})
    wb_cfg['enabled'] = False  # Don't init again in Crnn.py

    # Update paths to avoid conflicts between parallel runs
    cfg['paths']['out_dir'] = os.path.join(
        cfg['paths'].get('out_dir', './checkpoints'),
        f'sweep_{run.id}'
    )

    try:
        print(f"[SWEEP] Starting training...")
        Crnn.train_and_eval(cfg)
        print(f"[SWEEP] Training completed successfully for run {run.name}")
    except Exception as e:
        print(f"[SWEEP] ERROR in run {run.name}: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({"error": str(e), "error_type": type(e).__name__})
        raise
    finally:
        wandb.finish()


def init_sweep(config_path='config.yaml', project=None, entity=None):
    """Initialize a W&B sweep from your config.yaml"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    sweep_cfg = cfg.get('sweep', {})

    if not sweep_cfg:
        print("ERROR: No 'sweep' section found in config.yaml")
        sys.exit(1)

    if not sweep_cfg.get('enabled', False):
        print("WARNING: sweep.enabled is false in config.yaml")
        sweep_cfg['enabled'] = True

    # Build W&B sweep config
    sweep_config = {
        'method': sweep_cfg.get('method', 'bayes'),
        'metric': sweep_cfg.get('metric', {
            'name': 'val/f1',
            'goal': 'maximize'
        }),
        'parameters': sweep_cfg.get('parameters', {}),
        'program': 'sweepTrainer.py',
    }

    # Get project/entity
    wb_cfg = cfg.get('logging', {}).get('wandb', {})
    project = project or wb_cfg.get('project', 'CRNN-sweep')
    entity = entity or wb_cfg.get('entity')

    if not entity:
        print("ERROR: No W&B entity specified!")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Initializing W&B Sweep")
    print(f"{'=' * 60}")
    print(f"Project: {project}")
    print(f"Entity: {entity}")
    print(f"Method: {sweep_config['method']}")
    print(f"Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"\nParameters to sweep:")
    for param, config in sweep_config['parameters'].items():
        if 'values' in config:
            print(f"  {param}: {config['values']}")
        elif 'min' in config and 'max' in config:
            print(f"  {param}: [{config['min']}, {config['max']}]")
    print(f"{'=' * 60}\n")

    try:
        sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    except Exception as e:
        print(f"ERROR creating sweep: {e}")
        print("\nMake sure you're logged in: wandb login")
        sys.exit(1)

    print(f"✓ Sweep created successfully!")
    print(f"\nSweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print(f"\nTo start agents:")
    print(f"  wandb agent {entity}/{project}/{sweep_id}")

    return sweep_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true', help='Initialize a new sweep')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--entity', type=str, default=None)

    # Accept all W&B sweep parameters (but ignore them - we use wandb.config instead)
    parser.add_argument('--loss.focal.alpha', type=float, default=None)
    parser.add_argument('--loss.focal.gamma', type=float, default=None)
    parser.add_argument('--loss.name', type=str, default=None)
    parser.add_argument('--loss.weighted_bce.adaptive', type=str, default=None)
    parser.add_argument('--loss.weighted_bce.pos_weight', type=str, default=None)
    parser.add_argument('--model.fpn_ch', type=int, default=None)
    parser.add_argument('--model.rnn_hidden', type=int, default=None)
    parser.add_argument('--model.rnn_layers', type=int, default=None)
    parser.add_argument('--model.use_se', type=str, default=None)
    parser.add_argument('--trainer.batch_size', type=int, default=None)
    parser.add_argument('--trainer.lr', type=float, default=None)
    parser.add_argument('--trainer.sampler', type=str, default=None)
    parser.add_argument('--trainer.weight_decay', type=float, default=None)

    args, unknown = parser.parse_known_args()  # Ignore unknown args

    os.environ['CRNN_CONFIG_PATH'] = args.config

    if args.init:
        sweep_id = init_sweep(args.config, args.project, args.entity)
        return

    # Called by wandb agent
    train()


if __name__ == '__main__':
    main()