# main.py (simple & lowercase config)
import os
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from data_loader import make_loaders
from models_registry import make_model
from trainer import SpindleTrainer, TrainConfig
from metrics import SampleMetrics


def _cfg(cfg: Any, path: str, default=None):
    """Safe nested getter: _cfg(cfg, 'trainer.lr', 1e-3)."""
    cur = cfg
    for k in path.split("."):
        try:
            cur = cur[k] if isinstance(cur, dict) else getattr(cur, k)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _maybe_set_wandb_mode(mode: str | None):
    if not mode:
        return
    m = str(mode).lower()
    if m == "offline":
        os.environ["WANDB_MODE"] = "offline"
    elif m == "disabled":
        os.environ["WANDB_DISABLED"] = "true"


def _build_train_config(cfg: DictConfig) -> TrainConfig:
    return TrainConfig(
        DEVICE=_cfg(cfg, "trainer.device", "cuda" if torch.cuda.is_available() else "cpu"),
        EPOCHS=int(_cfg(cfg, "trainer.epochs", 50)),
        LR=float(_cfg(cfg, "trainer.lr", 1e-3)),
        BATCH_SIZE=int(_cfg(cfg, "trainer.batch_size", _cfg(cfg, "data.batch_size", 16))),
        WEIGHT_DECAY=float(_cfg(cfg, "trainer.weight_decay", 0.0)),
        OPTIMIZER=str(_cfg(cfg, "trainer.optimizer", "adam")),
        LOSS=str(_cfg(cfg, "trainer.loss", "bce")),
        EARLY_STOPPING_PATIENCE=int(_cfg(cfg, "trainer.early_stopping_patience", 10)),
        GRAD_CLIP_NORM=float(_cfg(cfg, "trainer.grad_clip_norm", 0.0)),
        USE_WANDB=bool(_cfg(cfg, "trainer.use_wandb", False)),
        WANDB_PROJECT=str(_cfg(cfg, "trainer.wandb_project", "eeg-spindle-detection")),
        MODEL_NAME=str(_cfg(cfg, "trainer.model_name", _cfg(cfg, "model.name", "unet1d"))),
        RUN_NAME=_cfg(cfg, "trainer.run_name", None),
    )


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    _maybe_set_wandb_mode(_cfg(cfg, "logging.wandb_mode", None))

    print("======== config (resolved) ========")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # 1) data
    train_loader, val_loader, test_loader = make_loaders(cfg)

    # 2) model
    model_name = str(_cfg(cfg, "model.name", "unet1d"))
    model_kwargs: Dict[str, Any] = dict(cfg.model)
    model_kwargs.pop("name", None)
    model = make_model(model_name, **model_kwargs)

    # 3) trainer
    tcfg = _build_train_config(cfg)
    trainer = SpindleTrainer(config=tcfg)
    extra_cfg = OmegaConf.to_container(cfg, resolve=True)

    # 4) train
    print("\n======== training ========")
    train_out = trainer.fit(model=model, train_loader=train_loader, val_loader=val_loader, extra_cfg=extra_cfg)
    print("training finished:", train_out)

    # 5) evaluation (pick threshold on VAL if requested)
    print("\n======== evaluation (test) ========")
    sm = SampleMetrics(
        sfreq=float(_cfg(cfg, "data.sfreq", 200.0)),
        window_sec=float(_cfg(cfg, "data.window_sec", 2.0)),
        step_sec=float(_cfg(cfg, "data.step_sec", 1.0)),
        use_wandb=bool(_cfg(cfg, "logging.use_wandb", False)),
        out_dir=str(_cfg(cfg, "paths.save_dir", "./_artifacts")),
    )
    device = str(_cfg(cfg, "trainer.device", "cpu"))

    # choose threshold
    thr = float(_cfg(cfg, "eval.threshold", 0.5))
    if bool(_cfg(cfg, "eval.use_val_best_threshold", True)) and (val_loader is not None):
        v_probs, v_labels = sm.stitch(model, val_loader, device=device)
        thr, best_conf = sm.best_threshold_from_arrays(v_labels, v_probs, num=101)
        print(f"val-chosen threshold: {thr:.4f}  val best F1={best_conf['f1']:.4f}")

    # evaluate on TEST once at chosen threshold
    eval_out = sm.evaluate(
        model=model,
        loader=test_loader,
        device=device,
        threshold=float(thr),
        sweep_threshold=False,
        log_curves=bool(_cfg(cfg, "eval.log_curves", False)),
    )

    print("\n===== test metrics =====")
    print(f"threshold: {eval_out.get('threshold', 0.0):.4f}")
    print(f"confusion: {eval_out.get('confusion')}")
    if "roc_auc" in eval_out:
        print(f"roc_auc: {eval_out['roc_auc']:.4f}")
    if "pr_auc" in eval_out:
        print(f"pr_auc: {eval_out['pr_auc']:.4f}")

    # 6) artifacts
    print("\n======== export artifacts ========")
    paths = sm.export_artifacts(
        model=model,
        loader=test_loader,
        cfg=cfg,
        device=device,
        prefix="test",
        save_dir=str(_cfg(cfg, "paths.save_dir", "./_artifacts")),
        save_arrays=True,
        save_json=True,
        save_plots=True,
    )
    for k, v in paths.items():
        print(f"{k}: {v}")

    # 7) finish wandb
    if bool(_cfg(cfg, "logging.use_wandb", False)) and "WANDB_DISABLED" not in os.environ:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
