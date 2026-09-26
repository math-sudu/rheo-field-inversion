"""Multi-branch neural continuation with FE optimality residuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class MultiBranchContinuationNet(nn.Module):
    """Map one profiled coordinate and a branch id to nuisance coordinates."""

    def __init__(
        self,
        n_outputs: int,
        n_branches: int = 2,
        width: int = 64,
        depth: int = 4,
        n_fourier: int = 4,
        embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.n_outputs = int(n_outputs)
        self.n_branches = int(n_branches)
        self.n_fourier = int(n_fourier)
        self.embedding = nn.Embedding(self.n_branches, embedding_dim)
        in_features = 1 + 2 * self.n_fourier + embedding_dim
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(in_features, width), nn.SiLU()])
            in_features = width
        layers.append(nn.Linear(in_features, self.n_outputs))
        self.body = nn.Sequential(*layers)

    def features(self, x: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        cols = [x]
        for k in range(1, self.n_fourier + 1):
            cols.extend([torch.sin(k * torch.pi * x),
                         torch.cos(k * torch.pi * x)])
        cols.append(self.embedding(branch))
        return torch.cat(cols, dim=1)

    def forward(self, x: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.body(self.features(x, branch)))


@dataclass(frozen=True)
class TrainingResult:
    model: MultiBranchContinuationNet
    history: list[dict[str, float]]
    final: dict[str, float]


def _component_derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    cols = []
    for j in range(y.shape[1]):
        grad = torch.autograd.grad(
            y[:, j].sum(), x, create_graph=True, retain_graph=True
        )[0]
        cols.append(grad)
    return torch.cat(cols, dim=1)


def train_continuation(
    *,
    x: np.ndarray,
    branch: np.ndarray,
    target: np.ndarray,
    correction: np.ndarray | None = None,
    gradient: np.ndarray,
    hessian: np.ndarray,
    tangent: np.ndarray,
    seed: int,
    use_physics: bool,
    epochs: int = 4000,
    learning_rate: float = 2.0e-3,
    data_weight: float = 1.0,
    correction_weight: float = 0.0,
    stationarity_weight: float = 0.25,
    tangent_weight: float = 0.25,
    device: str | None = None,
) -> TrainingResult:
    """Train with supervised anchors and FE-derived continuation equations.

    ``gradient`` and ``hessian`` are the exact-FE Gauss--Newton local
    optimality quantities in box-unit nuisance coordinates.  Stationarity is
    imposed through the unit-box projected-gradient mapping, not the invalid
    unconstrained condition at active bounds.  ``tangent`` is the fixed-active-
    set continuation derivative with respect to the network input x in [-1, 1].
    """

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    x_t = torch.as_tensor(x, dtype=dtype, device=device).reshape(-1, 1)
    branch_t = torch.as_tensor(branch, dtype=torch.long, device=device)
    target_t = torch.as_tensor(target, dtype=dtype, device=device)
    if correction is None:
        correction = np.zeros_like(target)
    correction_t = torch.as_tensor(correction, dtype=dtype, device=device)
    corrected_target_t = torch.clamp(target_t + correction_t, 0.0, 1.0)
    gradient_t = torch.as_tensor(gradient, dtype=dtype, device=device)
    hessian_t = torch.as_tensor(hessian, dtype=dtype, device=device)
    tangent_t = torch.as_tensor(tangent, dtype=dtype, device=device)
    model = MultiBranchContinuationNet(
        n_outputs=target_t.shape[1],
        n_branches=int(branch_t.max().item()) + 1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1.0e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=learning_rate * 0.02
    )
    history: list[dict[str, float]] = []
    final: dict[str, float] = {}
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        x_epoch = x_t.detach().clone().requires_grad_(use_physics)
        pred = model(x_epoch, branch_t)
        data_loss = torch.mean((pred - target_t) ** 2)
        correction_loss = torch.zeros((), dtype=dtype, device=device)
        stationarity_loss = torch.zeros((), dtype=dtype, device=device)
        tangent_loss = torch.zeros((), dtype=dtype, device=device)
        if use_physics:
            correction_loss = torch.mean((pred - corrected_target_t) ** 2)
            delta = (pred - target_t).unsqueeze(-1)
            stat = gradient_t.unsqueeze(-1) + torch.bmm(hessian_t, delta)
            scale = torch.linalg.matrix_norm(hessian_t, dim=(1, 2)).clamp_min(1.0)
            stat_scaled = stat.squeeze(-1) / scale[:, None]
            projected_step = pred - torch.clamp(pred - stat_scaled, 0.0, 1.0)
            stationarity_loss = torch.mean(projected_step ** 2)
            dz_dx = _component_derivative(pred, x_epoch)
            tangent_scale = (1.0 + torch.abs(tangent_t)).detach()
            tangent_loss = torch.mean(((dz_dx - tangent_t) / tangent_scale) ** 2)
        loss = (data_weight * data_loss
                + correction_weight * correction_loss
                + stationarity_weight * stationarity_loss
                + tangent_weight * tangent_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        final = {
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "correction_loss": float(correction_loss.detach().cpu()),
            "stationarity_loss": float(stationarity_loss.detach().cpu()),
            "tangent_loss": float(tangent_loss.detach().cpu()),
        }
        if epoch == 0 or (epoch + 1) % 250 == 0 or epoch + 1 == epochs:
            history.append({"epoch": float(epoch + 1), **final})
    return TrainingResult(model=model, history=history, final=final)


def predict_unit(
    model: MultiBranchContinuationNet,
    x: np.ndarray,
    branch: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)
        b_t = torch.as_tensor(branch, dtype=torch.long, device=device)
        return model(x_t, b_t).cpu().numpy()


def save_model(path: str | Path, result: TrainingResult, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "n_outputs": result.model.n_outputs,
            "n_branches": result.model.n_branches,
            "metadata": metadata,
            "history": result.history,
            "final": result.final,
        },
        path,
    )
