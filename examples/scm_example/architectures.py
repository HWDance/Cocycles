import torch
from torch import nn
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.transforms import (
    MonotonicAffineTransform,
    MonotonicRQSTransform,
)

def get_nsf_transforms():

    # Defining NSFs and MAFs (one per counterfactual state for independent init)
    nsf0 = [MaskedAutoregressiveTransform(
                features=1,
                context=0,
                hidden_features=(),
                univariate=MonotonicAffineTransform,
                shapes=[(), ()],
            ),
            MaskedAutoregressiveTransform(
                features=1,
                context=0,
                hidden_features=(),
                univariate=MonotonicRQSTransform,
                shapes=[(8,), (8,), (9,)],
            ),
           MaskedAutoregressiveTransform(
                features=1,
                context=0,
                hidden_features=(),
                univariate=MonotonicAffineTransform,
                shapes=[(), ()],
            ),
        ]
    nsf1 = [MaskedAutoregressiveTransform(
                features=1,
                context=0,
                hidden_features=(),
                univariate=MonotonicAffineTransform,
                shapes=[(), ()],
            ),
            MaskedAutoregressiveTransform(
                features=1,
                context=0,
                hidden_features=(),
                univariate=MonotonicRQSTransform,
                shapes=[(8,), (8,), (9,)],
            ),
           MaskedAutoregressiveTransform(
                features=1,
                context=0,
                hidden_features=(),
                univariate=MonotonicAffineTransform,
                shapes=[(), ()],
            ),
        ]
    return nsf0,nsf1

def get_maf_transforms():
    
    maf0 = [MaskedAutoregressiveTransform(
                features=1,
                context=0,
                hidden_features=(),
                univariate=MonotonicAffineTransform,
                shapes=[(), ()],
            ),
        ]
    maf1 = [MaskedAutoregressiveTransform(
                features=1,
                context=0,
                hidden_features=(),
                univariate=MonotonicAffineTransform,
                shapes=[(), ()],
            ),
        ]
    
    return maf0, maf1

# ── Discrete transforms for Cocycle  ─────────────────────────────────────────────────────
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

                if isinstance(flow_k, IdentityTransform):
                    y_k = flow_k.inverse(u_k)
                else:
                    # MaskedAutoregressiveTransform(context=0) → AutoregressiveTransform
                    ctx = u_k.new_empty((u_k.shape[0], 0))
                    bij = flow_k(None)            # AutoregressiveTransform
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

                if isinstance(flow_k, IdentityTransform):
                    u_k= flow_k.forward(y_k)
                else:
                    # MaskedAutoregressiveTransform(context=0) → AutoregressiveTransform
                    ctx = y_k.new_empty((y_k.shape[0], 0))
                    bij = flow_k(None)         # AutoregressiveTransform
                    # Use __call__ (i.e. bij(y_k)) to map data → latent
                    u_k = bij(y_k)            # returns a single Tensor

                u_out[mask] = u_k

            return u_out

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            """
            Alias so that `transform(u)` invokes `forward(u)`.
            """
            return self.forward(tensor)