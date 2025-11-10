import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def _normalize_adj(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    A = torch.clamp(A, min=0.0)
    eye = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    A = A + eps * eye
    deg = A.sum(dim=-1)
    inv_sqrt = torch.rsqrt(torch.clamp(deg, min=eps)).unsqueeze(-1)
    return inv_sqrt * A * inv_sqrt.transpose(-1, -2)


# -----------------------------
# Core layers
# -----------------------------

class TemporalDepthwiseConv(nn.Module):
    def __init__(self, F_in: int, F_out: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(F_in, F_out, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(F_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F_in)
        B, C, T, F_in = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B * C, F_in, T)
        x = self.conv(x); x = self.bn(x); x = self.act(x); x = self.drop(x)
        T2 = x.size(-1)
        x = x.view(B, C, -1, T2).permute(0, 1, 3, 2).contiguous()
        return x


class SqueezeExciteChannels(nn.Module):
    def __init__(self, C: int, r: int = 4):
        super().__init__()
        hidden = max(1, C // r)
        self.fc1 = nn.Linear(C, hidden)
        self.fc2 = nn.Linear(hidden, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F)
        s = x.mean(dim=(2, 3))
        w = torch.relu(self.fc1(s))
        w = torch.sigmoid(self.fc2(w)).view(x.size(0), x.size(1), 1, 1)
        return x * w


class AdaptiveGraphConv(nn.Module):
    def __init__(
        self,
        C: int,
        F_in: int,
        F_out: int,
        A_prior: Optional[torch.Tensor] = None,
        emb_dim: int = 8,
        lambda_init: float = 0.0,
        use_dynamic: bool = False,
        beta_init: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.C = C
        self.F_in = F_in
        self.F_out = F_out
        self.use_dynamic = use_dynamic

        if A_prior is None:
            A_prior = torch.eye(C)
        self.register_buffer('A_prior_raw', A_prior.float(), persistent=False)

        self.node_emb = nn.Parameter(torch.randn(C, emb_dim) * 0.1)
        self.lambda_logit = nn.Parameter(torch.logit(torch.tensor(min(max(lambda_init, 0.0), 1.0) + 1e-6)))
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor(min(max(beta_init, 0.0), 1.0) + 1e-6))) if use_dynamic else None

        self.lin = nn.Linear(F_in, F_out)
        self.bn = nn.BatchNorm2d(C)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_dynamic:
            d_k = max(8, F_in // 2)
            self.q_proj = nn.Linear(F_in, d_k)
            self.k_proj = nn.Linear(F_in, d_k)

    def _learned_adj(self) -> torch.Tensor:
        E = self.node_emb
        return F.softplus(E @ E.t())

    def _dynamic_adj(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F_in = x.shape
        q = self.q_proj(x).mean(dim=2)  # (B, C, d_k)
        k = self.k_proj(x).mean(dim=2)  # (B, C, d_k)
        att = torch.einsum('bcd,bed->bce', q, k) / math.sqrt(q.size(-1))
        A_dyn = F.relu(att)
        return _normalize_adj(A_dyn)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T, F_in)
        A_prior = _normalize_adj(self.A_prior_raw.to(x.device))
        A_learned = _normalize_adj(self._learned_adj())
        lam = torch.sigmoid(self.lambda_logit)
        A_star = (1.0 - lam) * A_prior + lam * A_learned  # (C, C)

        if self.use_dynamic:
            beta = torch.sigmoid(self.beta_logit)
            A_dyn = self._dynamic_adj(x)                  # (B, C, C)
            A_use = (1.0 - beta) * A_star.unsqueeze(0) + beta * A_dyn
        else:
            A_dyn = None
            A_use = A_star.unsqueeze(0)

        # aggregate neighbors per time
        Z = torch.einsum('bij, bjtf -> bitf', A_use, x)
        Z = self.lin(Z)
        Z = self.bn(Z)
        Z = self.act(Z)
        Z = self.drop(Z)
        return Z, A_prior, A_learned, A_dyn
# need to check if a_dyn is really i am using it ..... do research on diff graph

class STGCNBlock(nn.Module):
    def __init__(
        self,
        C: int,
        F_in: int,
        F_mid: int,
        F_out: int,
        A_prior: Optional[torch.Tensor] = None,
        dilation: int = 1,
        emb_dim: int = 8,
        lambda_init: float = 0.0,
        use_dynamic: bool = False,
        beta_init: float = 0.0,
        dropout: float = 0.0,
        se_ratio: int = 4,
    ):
        super().__init__()
        self.temporal = TemporalDepthwiseConv(F_in, F_mid, dilation=dilation, dropout=dropout)
        self.graph = AdaptiveGraphConv(
            C=C,
            F_in=F_mid,
            F_out=F_out,
            A_prior=A_prior,
            emb_dim=emb_dim,
            lambda_init=lambda_init,
            use_dynamic=use_dynamic,
            beta_init=beta_init,
            dropout=dropout,
        )
        self.se = SqueezeExciteChannels(C, r=se_ratio)
        self.res_proj = nn.Sequential(nn.Linear(F_in, F_out), nn.GELU()) if F_in != F_out else None

    def forward(self, x: torch.Tensor):
        z = self.temporal(x)
        z, A_p, A_l, A_d = self.graph(z)
        z = self.se(z)
        x_res = self.res_proj(x) if self.res_proj is not None else x
        return z + x_res, A_p, A_l, A_d


class STGCNSpindle(nn.Module):
    def __init__(
        self,
        C: int,
        T: int,
        F0: int = 16,
        block_channels: Tuple[int, int, int] = (32, 48, 64),
        dilations: Tuple[int, int, int] = (1, 2, 4),
        A_prior: Optional[torch.Tensor] = None,
        emb_dim: int = 8,
        lambda_init: float = 0.0,
        use_dynamic: bool = False,
        beta_init: float = 0.0,
        dropout: float = 0.1,
        se_ratio: int = 4,
    ):
        super().__init__()
        self.C, self.T = C, T
        self.A_prior = A_prior

        # front-end per-channel convs
        self.front = nn.Sequential(
            nn.Conv1d(1, F0, kernel_size=5, padding=2),
            nn.BatchNorm1d(F0),
            nn.GELU(),
            nn.Conv1d(F0, F0, kernel_size=3, padding=1),
            nn.BatchNorm1d(F0),
            nn.GELU(),
        )

        F_in = F0
        blocks = []
        for i, (F_mid, dil) in enumerate(zip(block_channels, dilations)):
            blk = STGCNBlock(
                C=C,
                F_in=F_in,
                F_mid=F_mid,
                F_out=block_channels[i],
                A_prior=A_prior,
                dilation=dil,
                emb_dim=emb_dim,
                lambda_init=lambda_init if i == 0 else 0.5,
                use_dynamic=use_dynamic and (i >= 1),
                beta_init=beta_init,
                dropout=dropout,
                se_ratio=se_ratio,
            )
            blocks.append(blk)
            F_in = block_channels[i]
        self.blocks = nn.ModuleList(blocks)
        self.F_last = F_in

        self.head_per_ch = nn.Linear(self.F_last, 1)
        self.chan_attn = nn.Sequential(
            nn.Linear(self.F_last, self.F_last // 2), nn.GELU(), nn.Linear(self.F_last // 2, 1)
        )
        self.head_global = nn.Linear(self.F_last, 1)
        self.head_window = nn.Linear(self.F_last, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, T = x.shape
        assert C == self.C and T == self.T, f"Expected (C,T)=({self.C},{self.T}), got ({C},{T})"

        # front-end
        x = x.unsqueeze(2).view(B * C, 1, T)
        x = self.front(x)
        x = x.view(B, C, -1, T).permute(0, 1, 3, 2).contiguous()  # (B,C,T,F)

        A_p = A_l = A_d_last = None
        for blk in self.blocks:
            x, A_p, A_l, A_d = blk(x)
            if A_d is not None:
                A_d_last = A_d

        logits_per_ch = self.head_per_ch(x).squeeze(-1)  # (B,C,T)

        attn_scores = self.chan_attn(x).squeeze(-1).transpose(1, 2)  # (B,T,C)
        attn_w = F.softmax(attn_scores, dim=-1)
        context = torch.einsum('btc,btcf->btf', attn_w, x.transpose(1, 2))  # (B,T,F)
        logits_global = self.head_global(context).squeeze(-1)              # (B,T)

        h_win = x.mean(dim=(1, 2))
        logit_window = self.head_window(h_win).squeeze(-1)

        out = {
            'logits_global': logits_global,
            'logits_per_ch': logits_per_ch,
            'logit_window': logit_window,
            'A_prior': _normalize_adj(self.A_prior.to(x.device)) if self.A_prior is not None else torch.eye(self.C, device=x.device),
            'A_learned': _normalize_adj(self.blocks[0].graph._learned_adj()),
            'A_dynamic': A_d_last,
        }
        return out
