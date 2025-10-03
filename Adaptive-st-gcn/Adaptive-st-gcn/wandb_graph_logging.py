from typing import Optional, Dict

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

import torch
import matplotlib.pyplot as plt

"""
W&B helper to log adjacency heatmaps (A_prior, A_learned, A_dynamic[0]).
"""


def _heatmap_img(mat: torch.Tensor, title: str):
    fig = plt.figure()
    plt.imshow(mat.detach().cpu().numpy())
    plt.title(title)
    plt.xlabel('channels'); plt.ylabel('channels')
    plt.colorbar(fraction=0.046, pad=0.04)
    fig.canvas.draw()
    return wandb.Image(fig)


def log_graphs_to_wandb(outputs: Dict[str, torch.Tensor], step: Optional[int] = None, prefix: str = "stgcn"):
    if wandb is None:
        return
    imgs = {}
    if 'A_prior' in outputs and outputs['A_prior'] is not None:
        imgs[f'{prefix}/A_prior'] = _heatmap_img(outputs['A_prior'], 'A_prior')
    if 'A_learned' in outputs and outputs['A_learned'] is not None:
        imgs[f'{prefix}/A_learned'] = _heatmap_img(outputs['A_learned'], 'A_learned')
    if 'A_dynamic' in outputs and outputs['A_dynamic'] is not None:
        imgs[f'{prefix}/A_dynamic_0'] = _heatmap_img(outputs['A_dynamic'][0], 'A_dynamic[0]')
    if imgs:
        wandb.log(imgs, step=step)
