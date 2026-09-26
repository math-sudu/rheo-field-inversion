"""Field continuation with separate local-quadratic and profile-tangent sets.

An off-stationary FE center still defines a residual linearization and its
box-constrained quadratic stationarity equation. It does not define a smooth
stationary-profile derivative. Tangent samples are optional, never filled with
zeros and mislabeled as physical derivatives.
"""
import numpy as np
import torch

from experiments.fe_pinc.network import MultiBranchContinuationNet, _component_derivative


def losses(model, arrays):
    prediction = model(arrays["value_x"], arrays["value_branch"])
    value = ((prediction-arrays["value_target"]).square().mean(1)*arrays["value_weight"]).sum()/arrays["value_weight"].sum()
    predicted = model(arrays["physics_x"], arrays["physics_branch"])
    delta = predicted-arrays["physics_target"]
    gradient = arrays["gradient"]+torch.bmm(arrays["hessian"], delta.unsqueeze(-1)).squeeze(-1)
    scale = torch.linalg.matrix_norm(arrays["hessian"], dim=(1, 2)).clamp_min(1.)
    projected = predicted-torch.clamp(predicted-gradient/scale[:, None], 0., 1.)
    out = dict(data_loss=value, correction_loss=(predicted-arrays["corrected_target"]).square().mean(),
               stationarity_loss=projected.square().mean())
    if len(arrays["tangent_x"]):
        x = arrays["tangent_x"].detach().clone().requires_grad_()
        derivative = _component_derivative(model(x, arrays["tangent_branch"]), x)
        out["tangent_loss"] = ((derivative-arrays["tangent"])/(1+arrays["tangent"].abs())).square().mean()
    return out


def train(arrays, settings, seed):
    if not len(arrays["physics_x"]):
        raise ValueError("Field FE-PINC requires actual local-physics samples")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    tensors = {k: torch.as_tensor(v, dtype=torch.long if k.endswith("branch") else torch.float32,
                                 device=settings["device"]) for k, v in arrays.items()}
    for k in ("value_x", "physics_x", "tangent_x"):
        tensors[k] = tensors[k].reshape(-1, 1)
    model = MultiBranchContinuationNet(n_outputs=arrays["value_target"].shape[1],
        n_branches=int(np.max(arrays["value_branch"]))+1, **settings["architecture"]).to(settings["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"],
                                                          eta_min=settings["learning_rate"]*settings["minimum_lr_fraction"])
    history = []
    weights = {"data_loss": settings["data_weight"], "correction_loss": settings["correction_weight"],
               "stationarity_loss": settings["stationarity_weight"], "tangent_loss": settings["tangent_weight"]}
    for epoch in range(settings["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        parts = losses(model, tensors)
        total = sum(weights[k]*v for k, v in parts.items())
        if not torch.isfinite(total):
            raise ValueError("Nonfinite field training loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
        optimizer.step()
        scheduler.step()
        if epoch == 0 or (epoch+1) % 250 == 0 or epoch+1 == settings["epochs"]:
            history.append(dict(epoch=epoch+1, loss=float(total.detach()),
                                **{k: float(v.detach()) for k, v in parts.items()}))
    # Evaluate the saved, post-update state rather than the last pre-update loss.
    final = {k: float(v.detach()) for k, v in losses(model, tensors).items()}
    return model, history, final
