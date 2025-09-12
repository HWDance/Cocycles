"""
Module: stock_transforms.py
Defines a suite of Masked Autoregressive Flow (MAF) variants for cross-validation in causal flows or cocycles.
Each factory returns a conditional MAF layer with a different univariate bijector.
Variants:
  1) Linear shift-only transform
  2) NN shift-only transform
  3) NN monotonic affine (scale+shift)
  4) Rational Quadratic Spline (RQS)
  5) Deep Sigmoidal Flow (DSF)

Also implements custom ShiftTransform and DSFMonotonic univariate bijectors.
"""
import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.nn.functional import softplus, softmax
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.transforms import (
    MonotonicAffineTransform,
    MonotonicRQSTransform,
    MonotonicTransform,
    Transform,
)

class ShiftTransform(Transform):
    """
    Univariate shift-only bijector: f(y) = y + shift
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, shift: torch.Tensor):
        # shift: Tensor of shape (..., 1) matching the feature dimension
        super().__init__(cache_size=0)
        self.shift = shift

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        # forward: add the shift
        return x + self.shift

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        # inverse: subtract the shift
        return y - self.shift

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # derivative = 1 ⇒ log|1| = 0
        return torch.zeros_like(x)


def DSFMonotonic(K_dsf: int):
    def univariate(phi: Tensor) -> Transform:
        K = K_dsf
        def f(x: Tensor) -> Tensor:
            raw_alpha = phi[..., :K]
            raw_beta  = phi[..., K:2*K]
            gamma     = phi[..., 2*K:3*K]
            delta     = phi[..., 3*K:]
            alpha = softmax(raw_alpha, dim=-1)
            beta  = softplus(raw_beta)
            sigs = torch.sigmoid(beta * x + gamma)
            return torch.logit((alpha * sigs).sum(dim=-1, keepdim=True)) + delta
        return MonotonicTransform(f=f, phi=(phi,), bound=10.0, eps=1e-6)
    return univariate

# -------------------------------- get_stock_transforms --------------------------------
def get_stock_transforms(x_dim: int, y_dim: int,mask = None):
    """
    Returns a list of MaskedAutoregressiveTransform instances conditioned on x_dim,
    each mapping y <-> u with different univariate bijectors.
    K_dsf: number of mixture components for DSF.
    """
    hidden = [64, 64]
    transforms = []

    # 1) Shift-only (linear) => uses custom ShiftTransform
    transforms.append(
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=(),
            univariate=ShiftTransform,
            shapes=[()],  # loc only
        )
    )
    # 2) NN shift-only: shift = learned scalar via MonotonicAffineTransform with no scale
    transforms.append(
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden,
            univariate=ShiftTransform,
            shapes=[()],  # loc only
        )
    )

    # 3) NN monotonic affine: learn loc and scale
    transforms.append(
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden,
            univariate=MonotonicAffineTransform,
            shapes=[(), ()],       
        )
    )

    # 4) Rational Quadratic Spline
    transforms.append(
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden,
            univariate=MonotonicRQSTransform,
            shapes=[(8,), (8,), (9,)],  # widths, heights, derivatives
        )
    )


    return transforms
