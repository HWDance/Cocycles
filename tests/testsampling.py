import torch
from torch.distributions import Normal,MultivariateNormal, Independent, Laplace
from testarchitectures import get_stock_transforms
from zuko.flows import UnconditionalDistribution
from causalflows.flows import CausalFlow
from causal_cocycle.conditionalflow_helper import sample_cf
import copy

def sample_do(
    flow: CausalFlow,
    X: torch.Tensor,
    index: int,
    intervention_fn: callable,
    sample_shape: torch.Size = torch.Size([1]),
) -> torch.Tensor:
    """
    Interventional sampling under context X (do on context).
    """
    # Number of context samples to draw
    num_samples = sample_shape[0]
    N = X.shape[0]
    # Randomly select context indices with replacement
    idx = torch.randint(low=0, high=N, size=(num_samples,), device=X.device)
    X_sub = X[idx]
    # Apply intervention to selected context rows
    X_do = copy.deepcopy(X_sub)
    X_do[...,index] = intervention_fn(X_do[...,index])
    # Build distribution under intervened context
    dist = flow(X_do)
    # Sample latent noise and map to output (maintaining sample_shape)
    z = dist.base.sample()
    y = dist.transform.inv(z)
    return y

# Simple sanity check for sample_do and distribution expansion
if __name__ == '__main__':
    torch.manual_seed(0)
    # Context X: 10 rows, single context dimension
    N = 10
    X = torch.randn(N, 1)
    # Feature Y will be output dim of flow; set y_dim=1
    y_dim = 2

    # Build a trivial transform for test (e.g., MAF or affine)
    transforms = get_stock_transforms(x_dim=1, y_dim=y_dim, mask=None)
    transform = transforms[0]

    # Base distribution: 1-D event shape
    base = UnconditionalDistribution(
        MultivariateNormal,
        loc=torch.zeros(y_dim),
        covariance_matrix=torch.eye(y_dim),
        buffer=True
    )

    def base_fn(c):
        # ignore context c for unconditional base
        loc = torch.zeros(y_dim, device=c.device)
        scale = torch.ones(y_dim, device=c.device)
        return Independent(Laplace(loc, scale), reinterpreted_batch_ndims=1)
    
    base = base_fn

    # Construct causal flow
    flow = CausalFlow(transform=transform, base=base)

    # 1) Check base distribution expansion
    print(X.shape)
    dist = flow(X)
    print("Base batch_shape:", dist.base.batch_shape)
    print("Base event_shape:", dist.base.event_shape)
    # Sample latent once per context row
    z = dist.base.sample()
    print("Latent sample shape (no args):", z.shape)

    # 2) Test sample_do: no-op intervention (identity)
    def identity_fn(old):
        return old
    y_do = sample_do(flow, X, index=0, intervention_fn=identity_fn, sample_shape = torch.Size([10]))
    print("sample_do output shape:", y_do.shape)

    # 3) Test sample_cf: use Y_obs = y_do for simplicity
    y_cf = sample_cf(flow, X, y_do, index=0, intervention_fn=lambda old: old)
    print("sample_cf output shape:", y_cf.shape)
    # Check that y_cf differs by ~1 from y_do on average
    print("Mean difference y_cf - y_do:", (y_cf - y_do).mean().item())