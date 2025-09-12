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
    CouplingTransform
)
from zuko.lazy import UnconditionalTransform

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

# -------------------------------- get_stock_transforms --------------------------------
def get_stock_transforms(x_dim: int, y_dim: int, mask=None):
    """
    Returns a list of sequences of MaskedAutoregressiveTransform instances conditioned on x_dim,
    each mapping y <-> u with different univariate bijectors.

    Parameters:
        x_dim : int
            Dimension of conditioning input.
        y_dim : int
            Output dimension (number of target variables).
        mask : list[Tensor] or None
            Optional list of masks to enforce custom autoregressive structure.
            Each mask is a (input_features, output_features) binary tensor.
    """
    hidden = [32, 32]
    transforms = []

    # 1) Shift-only (linear) => uses custom ShiftTransform
    transforms.append([
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=(),
            univariate=ShiftTransform,
            shapes=[()],
            adjacency=mask
        )
    ])

    # 2) NN shift-only: shift = learned scalar
    transforms.append([
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden,
            univariate=ShiftTransform,
            shapes=[()],
            adjacency=mask
        )
    ])

    # 3) NN monotonic affine: learn loc and scale
    transforms.append([
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden,
            univariate=MonotonicAffineTransform,
            shapes=[(), ()],
            adjacency=mask
        )
    ])

    # 4) RQS spline + affine
    transforms.append([
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden,
            univariate=MonotonicAffineTransform,
            shapes=[(), ()],
            adjacency=mask
        ),
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden,
            univariate=MonotonicRQSTransform,
            shapes=[(8,), (8,), (9,)],
            adjacency=mask
        ),
        MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden,
            univariate=MonotonicAffineTransform,
            shapes=[(), ()],
            adjacency=mask
        ),
    ])

    return transforms

def get_stock_coupling_transforms(x_dim: int, y_dim: int, mask=None):
    """
    Returns a list of lists of LazyModules (UnconditionalTransform)
    which each produce a CouplingTransform leaving X alone and
    autoregressively transforming Y via a MAF.
    """
    hidden = [32, 32]
    transforms = []

    # split_mask: True for X (unchanged), False for Y (to transform)
    split_mask = torch.cat([
        torch.ones(x_dim,  dtype=torch.bool),
        torch.zeros(y_dim, dtype=torch.bool),
    ])

    def build_maf(univariate, shapes, hidden_features):
        return MaskedAutoregressiveTransform(
            features=y_dim,
            context=x_dim,
            hidden_features=hidden_features,
            univariate=univariate,
            shapes=shapes,
            adjacency=mask
        )

    # Helper to wrap a single (meta, mask) into a LazyModule
    def lazy_coupling(maf):
        return UnconditionalTransform(
            CouplingTransform,
            maf,             # meta callable
            split_mask,      # BoolTensor mask
            buffer=True      # register mask as buffer
        )

    # 1) shift-only linear
    maf1 = build_maf(ShiftTransform, [()], ())
    transforms.append([ lazy_coupling(maf1) ])

    # 2) NN shift-only
    maf2 = build_maf(ShiftTransform, [()], hidden)
    transforms.append([ lazy_coupling(maf2) ])

    # 3) NN monotonic affine
    maf3 = build_maf(MonotonicAffineTransform, [(), ()], hidden)
    transforms.append([ lazy_coupling(maf3) ])

    # 4) RQS spline sandwiched by affines (three separate coupling steps)
    maf4a = build_maf(MonotonicAffineTransform, [(), ()], hidden)
    maf4b = build_maf(MonotonicRQSTransform,    [(8,), (8,), (9,)], hidden)
    maf4c = build_maf(MonotonicAffineTransform, [(), ()], hidden)

    transforms.append([
        lazy_coupling(maf4a),
        lazy_coupling(maf4b),
        lazy_coupling(maf4c),
    ])

    return transforms

