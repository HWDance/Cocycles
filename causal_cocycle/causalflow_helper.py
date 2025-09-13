import copy
from typing import Callable
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torch.distributions import Normal, Laplace, StudentT, Independent
from torch import nn
import torch.nn.functional as F
from causalflows.flows import CausalFlow

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import (
    Normal, Laplace, StudentT, Independent
)

class LearnableNormal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.raw_loc   = nn.Parameter(torch.zeros(dim))
        self.raw_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, c=None):
        loc   = self.raw_loc
        scale = F.softplus(self.raw_scale)
        return Independent(Normal(loc, scale), 1)

    def sample(self, *args, **kwargs):
        return self.forward(None).sample(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        return self.forward(None).rsample(*args, **kwargs)

    def log_prob(self, x):
        return self.forward(None).log_prob(x)


class LearnableLaplace(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.raw_loc   = nn.Parameter(torch.zeros(dim))
        self.raw_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, c=None):
        loc   = self.raw_loc
        scale = F.softplus(self.raw_scale)
        return Independent(Laplace(loc, scale), 1)

    def sample(self, *args, **kwargs):
        return self.forward(None).sample(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        return self.forward(None).rsample(*args, **kwargs)

    def log_prob(self, x):
        return self.forward(None).log_prob(x)

class LearnableStudentT(nn.Module):
    def __init__(self, dim, init_df=3.0, min_df=1e-2):
        super().__init__()
        self.raw_df    = nn.Parameter(torch.log(torch.tensor(init_df)))
        self.raw_loc   = nn.Parameter(torch.zeros(dim))
        self.raw_scale = nn.Parameter(torch.zeros(dim))
        self.min_df    = min_df

    def forward(self, c=None):
        df    = F.softplus(self.raw_df) + self.min_df
        loc   = self.raw_loc
        scale = F.softplus(self.raw_scale)
        return Independent(StudentT(df=df, loc=loc, scale=scale), 1) 

    def rsample(self, sample_shape=torch.Size()):
        # Reparameterization: T = loc + scale * Z * sqrt(df / V)
        # where Z ~ N(0,1), V ~ Gamma(df/2, rate=1/2)
        df = F.softplus(self.raw_df) + self.min_df
        loc = self.raw_loc
        scale = F.softplus(self.raw_scale)

        # Extend sample_shape to event shape
        dim = loc.shape[0]
        shape = sample_shape + (dim,)

        # 1) sample standard normal Z
        Z = torch.randn(shape, device=loc.device)

        # 2) sample V ~ Gamma(df/2, rate=1/2)
        concentration = df / 2
        rate = 0.5
        # Gamma.rsample supports gradients w.r.t. concentration and rate
        V = Gamma(concentration, rate).rsample(sample_shape)
        V = V.unsqueeze(-1)  # shape: sample_shape + (1,)

        # 3) form Student-T
        return loc + scale * Z * torch.sqrt(df / V)

    def sample(self, *args, **kwargs):
        return self.rsample(*args, **kwargs)

    def log_prob(self, x):
        return self.forward(None).log_prob(x)


def select_and_train_flow(
    flows: list,
    X: torch.Tensor,
    train_fraction: float = 1.0,
    k_folds: int = 2,
    num_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = None,
    flow_lrs: list = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, d = X.shape

    # Train/test split
    perm = torch.randperm(N)
    n_train = int(train_fraction * N)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]

    # Evaluate NLL
    def eval_nll(model, X_eval):
        model.eval()
        with torch.no_grad():
            X_eval = X_eval.to(device)
            dist = model()
            return -dist.log_prob(X_eval).mean().item()

    # CV on training set: for each flow
    kf = KFold(n_splits=k_folds, shuffle=True)
    cv_scores = []
    for i, flow in enumerate(flows):
        fold_scores = []
        lr_used = flow_lrs[i] if flow_lrs is not None else lr
        for train_f, val_f in kf.split(X_train):
            model = copy.deepcopy(flow).to(device).train()
            loader = DataLoader(TensorDataset(X_train[train_f]),
                                batch_size=batch_size, shuffle=True)
            optim = torch.optim.Adam(model.parameters(), lr=lr_used)
            # train
            for _ in range(num_epochs):
                for (x_b,) in loader:
                    x_b = x_b.to(device)
                    dist = model()
                    loss = -dist.log_prob(x_b).mean()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
            # validate
            fold_scores.append(eval_nll(model, X_train[val_f]))
        cv_scores.append(sum(fold_scores)/len(fold_scores))

    # pick best flow
    best_idx = int(torch.argmin(torch.tensor(cv_scores)))
    best_flow = flows[best_idx]

    # retrain best on entire set
    best_flow = copy.deepcopy(best_flow).to(device).train()
    loader_all = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(best_flow.parameters(), lr=lr)
    for _ in range(num_epochs):
        for (x_b,) in loader_all:
            x_b = x_b.to(device)
            dist = best_flow()
            loss = -dist.log_prob(x_b).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

    # final test NLL
    test_nll = eval_nll(best_flow, X_test.to(device)) if train_fraction < 1.0 else None

    return best_flow, test_nll, best_idx, cv_scores


def sample_do(
    flow: CausalFlow,
    index: int,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_shape: torch.Size = torch.Size([1]),
) -> torch.Tensor:
    dist = flow()
    z = dist.base.sample(sample_shape)
    x = dist.transform.inv(z)
    old = x[..., index]
    new = intervention_fn(old)
    x_mod = x.clone(); x_mod[..., index] = new
    z_mod = dist.transform(x_mod)
    z_prime = z.clone(); z_prime[..., index] = z_mod[..., index]
    return dist.transform.inv(z_prime)

def sample_do_backdoor(
    flow: CausalFlow,
    index: int,
    Z: torch.tensor,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_shape: torch.Size = torch.Size([1]),
) -> torch.Tensor:
    """
    Uses empirical dist for all nodes prior to index
    """
    dist = flow()
    z = dist.base.sample(sample_shape)
    x = dist.transform.inv(z)
    idx = torch.randint(low=0, high=len(Z), size=(sample_shape[0],), device=Z.device)
    Z_resample = Z[idx]
    x[...,:index] = Z_resample[...,:index]
    old = x[..., index]
    new = intervention_fn(old)
    x_mod = x.clone(); x_mod[..., index] = new
    z_mod = dist.transform(x_mod)
    z_prime = z.clone(); z_prime[..., :index+1] = z_mod[..., :index+1]
    return dist.transform.inv(z_prime)


def sample_cf(
    flow: CausalFlow,
    x_obs: torch.Tensor,
    index: int,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    orig = x_obs[:, index]
    new  = intervention_fn(orig)
    return flow().compute_counterfactual(x_obs, index=index, value=new)