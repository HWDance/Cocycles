"""
Module: stock_transforms.py
Defines a suite of Masked Autoregressive Flow (MAF) variants for cross-validation in causal flows or cocycles.
Each factory returns a conditional MAF layer with a different univariate bijector.
Variants:
  1) Linear shift-only transform
  2) NN shift-only transform
  3) NN monotonic affine (scale+shift)
  4) Rational Quadratic Spline (RQS)
"""

import copy
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.nn.functional import softplus, softmax
from zuko.lazy import UnconditionalTransform
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.flows.neural import MNN
from zuko.transforms import (
    MonotonicAffineTransform,
    MonotonicRQSTransform,
    MonotonicTransform,
    Transform,
    ComposedTransform,
    CouplingTransform,
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

# ------------------------------------------------------------------------------
# (1) IdentityTransform for anchoring x=0 → identity
# ------------------------------------------------------------------------------
class IdentityTransform(nn.Module):
    """
    A no-op bijector: forward(y) = (y, 0); inverse(u) = (u, 0).
    """

    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor):
        batch = y.shape[0]
        return y

    def inverse(self, u: torch.Tensor):
        batch = u.shape[0]
        return u

import torch
import torch.nn as nn
from zuko.flows.autoregressive import MaskedAutoregressiveTransform

class ChainTransform(nn.Module):
    """
    Wraps a list of MaskedAutoregressiveTransform modules into a single bijector.
    Usage:
        # 1) Build once:
        chain = ChainTransform(nn.ModuleList([maf_layer1, maf_layer2, ...]))
        # 2) At runtime, for each batch:
        ctx = torch.empty((batch_size, 0), device=device)  # zero‐dim context
        bij = chain(ctx)      # store ctx internally, return self
        u   = bij(y)          # data→latent
        y2  = bij.inv(u)      # latent→data
    """

    def __init__(self, maf_layers: nn.ModuleList):
        super().__init__()
        self.maf_layers = nn.ModuleList(maf_layers)
        self._ctx = None

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Data → latent.  y has shape (batch, y_dim).
        For each MAF layer in order:
          bij = maf(self._ctx)
          u = bij(y)
        Returns final u of shape (batch, y_dim).
        """
        u = y
        for maf in self.maf_layers:
            # Each maf expects a context of shape (batch, 0)
            bij = maf(self._ctx)    # returns an AutoregressiveTransform
            u = bij(u)              # __call__(data) → latent
        return u

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """
        Latent → data.  u has shape (batch, y_dim).
        For each MAF layer in reverse order:
          bij = maf(self._ctx)
          y = bij.inv(u)
        Returns final y of shape (batch, y_dim).
        """
        y = u
        for maf in reversed(self.maf_layers):
            bij = maf(self._ctx)    # AutoregressiveTransform
            y = bij.inv(y)          # latent→data
        return y


# ------------------------------------------------------------------------------
# (2) DiscreteSelectorTransform with correct handling of MAF(context=0)
# ------------------------------------------------------------------------------
class DiscreteSelectorTransform(nn.Module):
    """
    Wraps K unconditional bijector "factories" (MaskedAutoregressiveTransform or IdentityTransform).
    Calling DiscreteSelectorTransform(x) returns a SelectedBijector. Then:
       - SelectedBijector.inv(u)  dispatches to flows[k].inverse(...)
       - SelectedBijector.forward(y) dispatches to flows[k].forward(...)
    For MaskedAutoregressiveTransform, we pass a dummy zero‐dim context tensor of shape (n_k, 0).
    """

    def __init__(self, flows: nn.ModuleList):
        """
        Args:
          flows: ModuleList of length K, each either:
                 • IdentityTransform()  (nn.Module), or
                 • MaskedAutoregressiveTransform(context=0, …)
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.K = len(flows)

    def __call__(self, x: torch.Tensor):
        """
        Returns a SelectedBijector for the batch of labels x.
        Args:
          x: Tensor of shape (batch, 1) or (batch,), integer labels in {0,…,K-1}.
        """
        return DiscreteSelectorTransform.SelectedBijector(self, x)

# In architectures.py (only the SelectedBijector inner class shown)

    class SelectedBijector:
        def __init__(self, parent: "DiscreteSelectorTransform", x: torch.Tensor):
            self.parent = parent
            self.x_flat = x.view(-1).long()  # shape: (batch,)

        def inv(self, u: torch.Tensor) -> torch.Tensor:
            """
            Latent → data. For each i with label k, apply flows[k].inverse(u[i]).
            """
            batch, y_dim = u.shape
            y_out = torch.zeros_like(u)

            for k in range(self.parent.K):
                mask = (self.x_flat == k)
                if not mask.any():
                    continue
                u_k = u[mask]  # (n_k, y_dim)
                flow_k = self.parent.flows[k]

                if isinstance(flow_k, (IdentityTransform, ChainTransform)):
                    y_k = flow_k.inverse(u_k)
                else:
                    # MaskedAutoregressiveTransform(context=0) → AutoregressiveTransform
                    ctx = u_k.new_empty((u_k.shape[0], 0))
                    bij = flow_k(ctx)            # AutoregressiveTransform
                    # Use bij.inverse(...) to invert latents → data
                    y_k = bij.inv(u_k)       # returns a single Tensor

                y_out[mask] = y_k

            return y_out

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            """
            Data → latent. For each i with label k, apply flows[k].forward(y[i]).
            """
            batch, y_dim = y.shape
            u_out = torch.zeros_like(y)

            for k in range(self.parent.K):
                mask = (self.x_flat == k)
                if not mask.any():
                    continue
                y_k = y[mask]  # (n_k, y_dim)
                flow_k = self.parent.flows[k]

                if isinstance(flow_k, (IdentityTransform,ChainTransform)):
                    u_k= flow_k.forward(y_k)
                else:
                    # MaskedAutoregressiveTransform(context=0) → AutoregressiveTransform
                    ctx = y_k.new_empty((y_k.shape[0], 0))
                    bij = flow_k(ctx)         # AutoregressiveTransform
                    # Use __call__ (i.e. bij(y_k)) to map data → latent
                    u_k = bij(y_k)            # returns a single Tensor

                u_out[mask] = u_k

            return u_out

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            """
            Alias so that `transform(u)` invokes `forward(u)`.
            """
            return self.forward(tensor)
    
def get_anchored_discrete_flows_single(y_dim: int = 2, hidden: tuple = (64,64), mask = None) -> DiscreteSelectorTransform:
    """
    Constructs three unconditional flows and wraps them in DiscreteSelectorTransform:
      - flow0 = IdentityTransform()   (anchors domain 0 → identity)
      - flow1, flow2 = MaskedAutoregressiveTransform(context=0, hidden=hidden_features,
                            univariate=MonotonicAffineTransform, shapes=[(), ()])
    Returns:
      A DiscreteSelectorTransform containing [flow0, flow1, flow2].
    """

    architectures = []
    
    # NN monotonic affine: hidden MLP, MonotonicAffineTransform
    maf_maf_nn = MaskedAutoregressiveTransform(
        features=y_dim,
        context=0,
        hidden_features=hidden,
        univariate=MonotonicAffineTransform,
        shapes=[(), ()],
        adjacency=mask
    )
    flows = nn.ModuleList([
        IdentityTransform(),
        maf_maf_nn,
        copy.deepcopy(maf_maf_nn)
    ])
    architectures.append(DiscreteSelectorTransform(flows))

    return architectures
