"""CAD FE-PINC training with distinct value and local-physics sample sets.

The historical network architecture and trainer stay in experiments.fe_pinc.
Saved candidate values supervise both numerical labels independently. Only
centers with applicable local neighborhoods supply GN physics and tangents.
"""
import numpy as np
import torch

from experiments.fe_pinc.network import (
    MultiBranchContinuationNet, TrainingResult, _component_derivative,
)


def continuation_losses(model, tensors):
    """Evaluate independently normalized value and local-physics losses."""
    value = model(tensors["value_x"], tensors["value_branch"])
    error = torch.mean((value-tensors["value_target"])**2, dim=1)
    weight = tensors["value_weight"]
    data = torch.sum(weight*error)/weight.sum()

    x = tensors["x"].detach().clone().requires_grad_()
    pred = model(x, tensors["branch"])
    target, hessian = tensors["target"], tensors["hessian"]
    corrected = torch.clamp(target+tensors["correction"], 0., 1.)
    stat = tensors["gradient"] + torch.bmm(hessian, (pred-target).unsqueeze(-1)).squeeze(-1)
    scale = torch.linalg.matrix_norm(hessian, dim=(1, 2)).clamp_min(1.)
    projected = pred-torch.clamp(pred-stat/scale[:, None], 0., 1.)
    tangent = tensors["tangent"]
    derivative = _component_derivative(pred, x)
    return {"data_loss": data, "correction_loss": torch.mean((pred-corrected)**2),
        "stationarity_loss": torch.mean(projected**2),
        "tangent_loss": torch.mean(((derivative-tangent)/(1+torch.abs(tangent)))**2)}


def train_continuation(*, value_x, value_branch, value_target, value_weight,
        x, branch, target, correction, gradient, hessian, tangent, seed,
        epochs=2000, learning_rate=.002, data_weight=1., correction_weight=1.,
        stationarity_weight=.25, tangent_weight=.0001, device=None):
    """Fit saved values while applying local equations only to their own rows.

Data weights give each original center and its available sides one unit of
total supervision. Local physics keeps the previous per-component mean and
the same GN normalization. No secant is substituted for a tangent target.
    """
    if int(epochs) < 1:
        raise ValueError("Training requires at least one epoch")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = dict(value_x=value_x, value_branch=value_branch, value_target=value_target,
        value_weight=value_weight, x=x, branch=branch, target=target,
        correction=correction, gradient=gradient, hessian=hessian, tangent=tangent)
    tensors = {name: torch.as_tensor(a, device=device,
        dtype=torch.long if name in ("branch", "value_branch") else torch.float32)
        for name, a in arrays.items()}
    for name in ("x", "value_x"):
        tensors[name] = tensors[name].reshape(-1, 1)
    model = MultiBranchContinuationNet(n_outputs=target.shape[1],
        n_branches=int(np.max(value_branch))+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs), eta_min=learning_rate*.02)
    weights = {"data_loss": data_weight, "correction_loss": correction_weight,
        "stationarity_loss": stationarity_weight, "tangent_loss": tangent_weight}
    history = []
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        losses = continuation_losses(model, tensors)
        loss = sum(weights[name]*value for name, value in losses.items())
        if not torch.isfinite(loss):
            raise ValueError("FE-PINC training produced a nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)
        optimizer.step()
        scheduler.step()
        final = {"loss": float(loss.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in losses.items()}}
        if epoch == 0 or (epoch+1) % 250 == 0 or epoch+1 == epochs:
            history.append({"epoch": float(epoch+1), **final})
    return TrainingResult(model=model, history=history, final=final)
