import csv
from typing import List

import torch

"""
Load 10–20 electrode coordinates and build an RBF distance prior adjacency.
Return A_prior (unnormalized). The ST-GCN model normalizes inside forward.
"""


def load_coords_from_csv(csv_path: str, channel_names: List[str], normalize: bool = True) -> torch.Tensor:
    """this will Load electrode coords and reorder rows to match channel_names.
    CSV columns: name,x,y,z  (z optional; if missing, set z=0)
    Returns: Tensor (C, D) with D=2 or 3
    """
    rows = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r['name'].strip()
            x = float(r['x']); y = float(r['y'])
            z = float(r['z']) if 'z' in r and r['z'] != '' else 0.0
            rows[name] = (x, y, z)

    coords = []
    for ch in channel_names:
        if ch not in rows:
            raise ValueError(f"Channel '{ch}' not found in coords CSV. Add it or fix channel_names.")
        coords.append(rows[ch])
    coords = torch.tensor(coords, dtype=torch.float32)  # (C,3)

    # drop z if all zeros
    if torch.allclose(coords[:, 2], torch.zeros_like(coords[:, 2])):
        coords = coords[:, :2]

    if normalize:
        coords = coords - coords.mean(dim=0, keepdim=True)
        max_abs = coords.abs().max().clamp(min=1e-6)
        coords = coords / max_abs
    return coords


def make_distance_prior(coords: torch.Tensor, sigma: float = 0.15) -> torch.Tensor:
    """RBF kernel on Euclidean distance. coords: (C, D) -> A: (C, C)
    """
    d2 = torch.cdist(coords, coords, p=2) ** 2
    A = torch.exp(-d2 / (2 * sigma * sigma))
    A.fill_diagonal_(1.0)
    return A


def build_distance_prior(csv_path: str, channel_names: List[str], sigma: float = 0.15, normalize_coords: bool = True) -> torch.Tensor:
    coords = load_coords_from_csv(csv_path, channel_names, normalize=normalize_coords)
    return make_distance_prior(coords, sigma=sigma)
