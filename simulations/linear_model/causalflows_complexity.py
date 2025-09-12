# Imports
import copy
import torch
from typing import Callable
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.transforms import Transform
from torch.distributions import constraints, Distribution, Normal, Laplace, Cauchy, Gamma, Uniform
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.flows import UnconditionalDistribution
from zuko.transforms import MonotonicRQSTransform
from causalflows.flows import CausalFlow
from sklearn.model_selection import KFold
from causal_cocycle.causalflow_helper import select_and_train_flow, sample_do, sample_cf
from causal_cocycle.helper_functions import ks_statistic, wasserstein1_repeat, rmse

class ShiftTransform(Transform):
    do
    main = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, b):
        # b will be a tensor of shape (batch,1)
        super().__init__(cache_size=0)
        self.b = b

    def _call(self, x):
        # x: (B,), b: (B,1) ⇒ squeeze to (B,)
        shift = self.b.squeeze(-1)
        return x + shift

    def _inverse(self, y):
        shift = self.b.squeeze(-1)
        return y - shift

    def log_abs_det_jacobian(self, x, y):
        # d(x+shift)/dx = 1  ⇒ log|1| = 0
        return torch.zeros_like(x)

"""
def select_and_train_flow(
    flows: list,
    X: torch.Tensor,
    train_fraction: float = 1.0,
    k_folds: int = 2,
    num_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = None,
    ):
    """
"""
    1) Split X into train/test
    2) For each flow in `flows`:
         • run k-fold CV on train set, computing mean held-out NLL
    3) Pick best flow (min mean NLL), fresh deepcopy its prototype
    4) Train that best flow on *all* of X (num_epochs, batch_size)
    5) Return (trained_flow, test_nll)
    """
"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N, d = X.shape
    # 1) train/test split
    perm = torch.randperm(N)
    n_train = int(train_fraction * N)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    
    # helper to compute NLL on a dataset
    def eval_nll(model, X_eval):
        model.eval()
        with torch.no_grad():
                X_eval = X_eval.to(device)
                dist = model()
                nll = -dist.log_prob(X_eval).mean().item()
        return nll  # average NLL per example
    
    # 2) CV on train set
    kf = KFold(n_splits=k_folds, shuffle=True)
    cv_scores = []
    
    for proto in flows:
        scores = []
        for train_f, val_f in kf.split(X_train):
            # fresh copy of the prototype
            model = copy.deepcopy(proto).to(device)
            model.train()
            # DataLoaders for this fold
            ds_tr = TensorDataset(X_train[train_f])
            ds_val = X_train[val_f]
            loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
            # train on this fold
            for _ in range(num_epochs):
                for (x_b,) in loader_tr:
                    x_b = x_b.to(device)
                    dist = model()
                    loss = -dist.log_prob(x_b).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
            # evaluate on the val fold
            nll_val = eval_nll(model, ds_val.to(device))
            scores.append(nll_val)
    
        cv_scores.append(sum(scores)/len(scores))
    
    # 3) pick best
    best_idx = int(torch.argmin(torch.tensor(cv_scores)))
    best_proto = flows[best_idx]
    
    # 4) retrain on *all* X
    best_flow = copy.deepcopy(best_proto).to(device)
    best_flow.train()
    ds_all = TensorDataset(X)
    loader_all = DataLoader(ds_all, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(best_flow.parameters(), lr=lr)
    for _ in range(num_epochs):
        for (x_b,) in loader_all:
            x_b = x_b.to(device)
            dist = best_flow()
            loss = -dist.log_prob(x_b).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # 5) final test evaluation
    if train_fraction < 1.0:
        test_nll = eval_nll(best_flow, X_test.to(device))
    else:
        test_nll = None
    
    return best_flow, test_nll, best_idx, cv_scores

"""
"""
Algorithm 1 from "Causal Normalizing Flows: From Theory to Practice,
    Adrian Javaloy, Pablo Sanchez-Martin, Isabel Valera, (2024)", generalized to transformation
    interventions
"""
"""
def sample_do(
    flow: CausalFlow,
    index: int,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_shape: torch.Size = torch.Size([1]),
) -> torch.Tensor:
    """
"""
    Algorithm 1 (streamlined):
      1) sample z ~ p(z)           (base)
      2) x = f^{-1}(z)              (decode)
      3) new_i = intervention_fn(x[..., index])
      4) x_mod = x with x_mod[...,index]=new_i
      5) z_mod = f(x_mod)           (encode)
      6) z_prime = z; z_prime[...,index]=z_mod[...,index]
      7) x_do = f^{-1}(z_prime)     (decode)
    """
"""
    # 1) & 2)
    dist = flow()                              # NormalizingFlow p(X|context)
    z = dist.base.sample(sample_shape)                # sample latent
    x = dist.transform.inv(z)                     # decode to data

    # 3) compute new intervention values
    old_i = x[..., index]
    new_i = intervention_fn(old_i)

    # 4) form the modified data batch
    x_mod = x.clone()
    x_mod[..., index] = new_i

    # 5) re-encode only the intervened batch
    z_mod = dist.transform(x_mod)

    # 6) splice in the new latent for feature `index`
    z_prime = z.clone()
    z_prime[..., index] = z_mod[..., index]

    # 7) decode the fully downstream‐consistent do‐sample
    x_do = dist.transform.inv(z_prime)
    return x_do

"""
"""
Algorithm 2 from "Causal Normalizing Flows: From Theory to Practice,
    Adrian Javaloy, Pablo Sanchez-Martin, Isabel Valera, (2024)", generalized to transformation
    interventions
    """
"""
def sample_cf(
    flow: CausalFlow,
    x_obs: torch.Tensor,
    index: int,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
"""
    Algorithm 2: compute counterfactual X_cf given factual x_obs under do(X[index] = intervention_fn).

    Args:
        flow: a fitted CausalFlow instance
        x_obs: factual observations, shape (batch, d)
        context: tensor of shape (batch, c_dim) or None if unconditional
        index: which feature to intervene on
        intervention_fn: a function f(old_vals) -> new_vals

    Returns:
        Counterfactuals, shape (batch, d)
    """
"""
    # 1) compute the per-example new value
    orig = x_obs[:, index]                    # shape = (batch,)
    new  = intervention_fn(orig)              # same shape

    # 2) use built‐in CF helper
    return flow().compute_counterfactual(
        x_obs,
        index=index,
        value=new,
    )

def ks_statistic(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten(); b = b.flatten()
    a_s, _ = torch.sort(a); b_s, _ = torch.sort(b)
    all_vs = torch.cat([a_s, b_s]).unique()
    cdf_a = torch.bucketize(all_vs, a_s, right=True).float() / a_s.numel()
    cdf_b = torch.bucketize(all_vs, b_s, right=True).float() / b_s.numel()
    return (torch.abs(cdf_a - cdf_b).max()).item()

def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(((a - b)**2).mean()).item()

"""
def evaluate_models(
    models_dict: dict,
    index_dict: dict,
    X: torch.Tensor,
    Y: torch.Tensor,
    noisedist: Distribution,
    noisetransform: callable,
    sig_noise_ratio: float,
    seed: int = None
) -> dict:
    """
    Adapted to work with a joint causal flow in models_dict.
    Expects models_dict = {'Flow': (flow, 'flow')},
             index_dict  = {'Flow': (idx, 'flow')}.
    Returns the same keys: KS_int, CF_RMSE, index under 'Flow'.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = X.device
    N, D = X.shape
    _, P = Y.shape
    assert D == 1 and P == 1  # 2D joint (X,Y)

    # “true” counterfactual shift ΔY = +1
    Z = torch.cat([X, Y], dim=1).to(device)             # shape (N,2)
    X_cf = X + 1.0
    ΔY_true = torch.ones((N, P), device=device)

    # “true” interventional Y distribution:
    # Y_true = X_cf + U    with U = noisetransform(noisedist.sample)
    m = 10**5
    # sample X* from Normal(1,1) then +1, and add noise:
    Y_true = (
        Normal(1, 1).sample((m, D)).to(device)
        + 1.0
        + noisetransform(noisedist.sample((m, 1)).to(device))
    )  # shape (m,1)

    results = {}
    for name, (flow, _) in models_dict.items():

        # ---- interventional estimate via Alg 1 ----
        # sample m draws from the *joint* under do(X = 1)
        Y_do = sample_do(
            flow.to(device),
            index=0,
            intervention_fn=lambda old: old + 1.0,
            sample_shape=torch.Size([m])
        )  # shape (m, 2)
        Y_int = Y_do[:, 1].unsqueeze(-1)  # shape (m,1)
        KS_int = ks_statistic(Y_int[:,0].cpu(), Y_true[:,0].cpu())

        # ---- counterfactual via Alg 2 ----
        Z_cf = sample_cf(
            flow.to(device),
            x_obs=Z,
            index=0,
            intervention_fn=lambda old: old + 1.0
        )  # shape (N,2)
        ΔY_model = (Z_cf[:,1] - Z[:,1]).unsqueeze(-1)  # (N,1)
        CF_RMSE = rmse(ΔY_model[:,0].cpu(), ΔY_true[:,0].cpu())

        results[name] = {
            'KS_int':  KS_int,
            'CF_RMSE': CF_RMSE,
            'index': index_dict[name][0]
        }

    # add noise info if you like (mirroring your old script)
    results['noise_distribution'] = noisedist.__class__.__name__
    return results

def run_experiment(seed=0, N=1000, noise_dist = "normal", lin_model = False):

    """
    Configs
    """
    # Experimental set up
    D,P = 1,1
    sig_noise_ratio = 1

    # Model setup
    width = 32
    bins = 8
    
    """
    Data gen
    """
    torch.manual_seed(seed)
    X = Normal(1,1).sample((N,D))
    X *= 1/(D)**0.5
    B = torch.ones((D,1))*(torch.linspace(0,D-1,D)<P)[:,None]
    F = X @ B
    if noise_dist == "normal":
        noisedist = Normal(0,1)
        noisetransform = lambda x : x
    elif noise_dist == "rademacher": 
        noisedist = Uniform(-1,1)
        noisetransform = lambda x : torch.sign(x)
    elif noise_dist == "cauchy":
        noisedist = Cauchy(0,1)
        noisetransform = lambda x : x
    elif noise_dist == "gamma":
        noisedist = Gamma(1,1)
        noisetransform = lambda x : x
    elif noise_dist == "inversegamma":
        noisedist = Gamma(1,1)
        noisetransform = lambda x : 1/x
    U = noisetransform(noisedist.sample((N,1)))/sig_noise_ratio**0.5
    Y = F + U
    XY = torch.cat([X,Y],dim = 1)

    """
    Defining list of flows to CV over
    """
    
    # shared base distribution
    baseg = UnconditionalDistribution(
        Normal,
        loc=torch.zeros(2),
        scale=torch.ones(2),
        buffer=True,
    )

    basel = UnconditionalDistribution(
        Laplace,
        loc=torch.zeros(2),
        scale=torch.ones(2),
        buffer=True,
    )
    
    
    # 1) MAF only, shift linear conditioner
    maf_shift_only = MaskedAutoregressiveTransform(
        features=2,
        context=0,
        hidden_features=(),   # you can also use () for purely linear shift
        univariate=ShiftTransform,  # use our shift‐only bijector
        shapes=([1],),              # one shift‐parameter per dimension
    )
    flow_shift_only_g = CausalFlow(
        transform=[maf_shift_only],
        base=baseg,
    )
    flow_shift_only_l = CausalFlow(
        transform=[maf_shift_only],
        base=basel,
    )
    
    # 3) MAF only, shift neural‑net conditioner
    maf_nn_shift_only = MaskedAutoregressiveTransform(
        features=2,
        context=0,
        hidden_features=(width, width),   # you can also use () for purely linear shift
        univariate=ShiftTransform,  # use our shift‐only bijector
        shapes=([1],),              # one shift‐parameter per dimension
    )
    flow_nn_shift_only_g = CausalFlow(
        transform=[maf_nn_shift_only],
        base=baseg,
    )
    flow_nn_shift_only_l = CausalFlow(
        transform=[maf_nn_shift_only],
        base=basel,
    )
    
    # 3) MAF only, neural‑net conditioner
    maf_nn = MaskedAutoregressiveTransform(
        features=2,
        context=0,
        hidden_features=(width, width),  # two hidden layers of size 32
    )
    flow_maf_nn_g = CausalFlow(transform=[maf_nn], base=baseg)
    flow_maf_nn_l = CausalFlow(transform=[maf_nn], base=basel)
    
    # 4) MAF → RQS, both neural‑net conditioners
    maf_nn = MaskedAutoregressiveTransform(
        features=2,
        context=0,
        hidden_features=(width, width),
    )
    rqs_nn = MaskedAutoregressiveTransform(
        features=2,
        context=0,
        hidden_features=(width, width),
        univariate=MonotonicRQSTransform,
        shapes=([bins], [bins], [bins + 1]),
    )
    flow_maf_rqs_nn_g = CausalFlow(transform=[maf_nn, rqs_nn], base=baseg)
    flow_maf_rqs_nn_l = CausalFlow(transform=[maf_nn, rqs_nn], base=basel)

    # Gaussian + Laplace base flows
    flows_g, flows_l = [flows_shift_only_g], [flows_shift_only_l]
    if not lin_model:
        flows_g += [
                flow_nn_shift_only_g,
                flow_maf_nn_g,
                flow_maf_rqs_nn_g
        ]
        flows_l += [
                flow_nn_shift_only_l,
                flow_maf_nn_l,
                flow_maf_rqs_nn_l
        ]


    """
    Training
    """
    # Gaussian‐base
    best_g, test_nll_g, idx_g, cv_scores_g = select_and_train_flow(
        flows_g,
        XY,
        train_fraction=1.0,
        k_folds=2,
        num_epochs=1000,
        batch_size=128,
        lr=1e-2,
        device=XY.device
    )

    # Laplace‐base
    best_l, test_nll_l, idx_l, cv_scores_l = select_and_train_flow(
        flows_l,
        XY,
        train_fraction=1.0,
        k_folds=2,
        num_epochs=1000,
        batch_size=128,
        lr=1e-2,
        device=XY.device
    )

    """
    Evaluating
    """
    # 6) Wrap into the same my_models / my_indexes format
    # --------------------------------------------------------------------------------
    my_models = {
        'GaussianFlow': (best_g, 'flow'),
        'LaplaceFlow':  (best_l, 'flow'),
    }
    my_indexes = {
        'GaussianFlow': (idx_g, 'flow'),
        'LaplaceFlow':  (idx_l, 'flow'),
    }

    # 7) Evaluate with your original evaluate_models
    # --------------------------------------------------------------------------------
    metrics = evaluate_models(
        my_models,
        my_indexes,
        X, Y,
        noisedist,
        noisetransform,
        sig_noise_ratio,
        seed=seed
    )

    # 8) Add test‐NLL and CV‐score diagnostics
    # --------------------------------------------------------------------------------
    metrics['test_nll_gaussian']   = test_nll_g
    metrics['test_nll_laplace']    = test_nll_l
    metrics['cv_scores_gaussian']  = cv_scores_g
    metrics['cv_scores_laplace']   = cv_scores_l

    return metrics
   

    