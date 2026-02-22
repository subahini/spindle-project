import argparse
import copy
import yaml
import wandb


def deep_set(cfg: dict, dotted_key: str, value):
    """
    Set cfg['trainer']['lr'] when dotted_key = 'trainer.lr'
    """
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def apply_wandb_overrides(base_cfg: dict, wb_cfg: dict) -> dict:
    """
    Apply wandb.config (flat dict) to nested YAML config using dotted keys.
    """
    cfg = copy.deepcopy(base_cfg)
    for k, v in wb_cfg.items():
        deep_set(cfg, k, v)
    return cfg


def train_one_run(cfg: dict):
    """
    Call YOUR existing training entry here.
    Example:
        from train import train
        train(cfg)
    """
    # --- IMPORTANT: replace this with your real train call ---
    print("TRAIN RUN with:")
    print("  trainer.lr =", cfg["trainer"]["lr"])
    print("  trainer.batch_size =", cfg["trainer"]["batch_size"])
    print("  loss.name =", cfg["loss"]["name"])
    print("  model.rnn_hidden =", cfg["model"]["rnn_hidden"])
    # --------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--create", action="store_true", help="Create sweep and print ID")
    parser.add_argument("--agent", type=str, default=None, help="Run agent for this sweep id")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(args.config, "r"))

    # W&B logging config
    wb = base_cfg.get("logging", {}).get("wandb", {})
    entity = base_cfg.get("project", {}).get("entity", wb.get("entity", None))
    project = wb.get("project", base_cfg.get("project", {}).get("name", "spindle_project"))

    sweep_cfg = base_cfg.get("sweep", None)
    if sweep_cfg is None or not sweep_cfg.get("enabled", False):
        raise ValueError("No sweep config found or sweep.enabled is false in config.yaml")

    # W&B expects this exact dict structure
    wandb_sweep_dict = {
        "method": sweep_cfg["method"],
        "metric": sweep_cfg["metric"],
        "parameters": sweep_cfg["parameters"],
    }

    if args.create:
        sweep_id = wandb.sweep(wandb_sweep_dict, project=project, entity=entity)
        print("SWEEP_ID:", sweep_id)
        return

    if args.agent is None:
        raise ValueError("Provide --agent <SWEEP_ID> or use --create")

    def _run():
        # wandb.config holds the sampled hyperparameters for this run
        run_cfg = apply_wandb_overrides(base_cfg, dict(wandb.config))

        wandb.init(
            project=project,
            entity=entity,
            config=run_cfg,  # logs full merged config
            tags=wb.get("tags", []),
        )

        train_one_run(run_cfg)
        wandb.finish()

    wandb.agent(args.agent, function=_run, project=project, entity=entity)


if __name__ == "__main__":
    main()