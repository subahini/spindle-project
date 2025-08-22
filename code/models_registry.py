# models_registry.py
from typing import Any, Dict, Type
from torch import nn
from models import UNet1D_Sample, TCN1D_Sample, CNN2D_Sample

_REG: Dict[str, tuple[Type[nn.Module], str]] = {
    "unet1d": (UNet1D_Sample, "BCT"),
    "tcn1d":  (TCN1D_Sample,  "BCT"),
    "cnn2d":  (CNN2D_Sample,  "B1CT"),
}

def make_model(name: str, **kwargs: Any) -> nn.Module:
    key = (name or "").lower()
    if key not in _REG:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REG.keys())}")
    cls, layout = _REG[key]
    m = cls(**kwargs)
    m._expected_input = layout  # tells trainer how to adapt [B,C,T]
    return m
