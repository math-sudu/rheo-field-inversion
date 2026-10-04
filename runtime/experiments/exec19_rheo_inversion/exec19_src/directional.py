"""Forward derivatives of the same implicit replay used by the adjoint.

The sparse inverse and its transposed adjoint are inherited unchanged from
the FE core. The JVP adds the action of that fixed inverse on a tangent RHS.
For the viscous lane, a batched sweep over the free coordinates avoids one
reverse sweep per observation without changing the estimator or FE history.
The CWFS lane retains its faster reverse mode; both modes are cross-locked.
"""
import numpy as np
import torch
from fedev_src.diff import _SpluSolve
from fedev_src.meshing import DTYPE


class _DirectionalSolve(torch.autograd.Function):
    @staticmethod
    def forward(rhs, lu):
        return torch.from_numpy(lu.solve(np.ascontiguousarray(rhs.detach().numpy())))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.lu = inputs[1]

    @staticmethod
    def backward(ctx, grad_out):
        return _SpluSolve.backward(ctx, grad_out)

    @staticmethod
    def jvp(ctx, rhs_tangent, lu_tangent):
        if rhs_tangent is None:
            return None
        return _DirectionalSolve.apply(rhs_tangent, ctx.lu)

    @staticmethod
    def vmap(info, in_dims, rhs, lu):
        if in_dims[0] is None:
            return _DirectionalSolve.apply(rhs, lu), None
        result = _DirectionalSolve.apply(rhs.movedim(in_dims[0], -1), lu)
        return result, result.ndim - 1


def lu_solve_t(lu, rhs):
    return _DirectionalSolve.apply(rhs, lu)


def jacobian_columns(evaluate, values):
    """Evaluate a vector and its exact AD columns at scalar coordinates."""
    names = tuple(values)
    initial = torch.tensor([float(values[n]) for n in names], dtype=DTYPE)
    def vector_function(vector):
        result = evaluate(dict(zip(names, vector.unbind())))
        return result, result
    with torch.no_grad():
        if not names:
            primal = evaluate({}).detach().numpy()
            return primal, np.empty((len(primal), 0))
        jacobian, primal = torch.func.jacfwd(vector_function, has_aux=True)(initial)
    return primal.detach().numpy(), jacobian.detach().numpy()
