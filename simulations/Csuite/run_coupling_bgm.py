import torch
from torch.distributions import (
    Distribution, Normal, Laplace, Cauchy, Gamma, Uniform
)
from csuite import SCMS, SCM_DIMS, SCM_MASKS
from architectures import get_stock_coupling_transforms
from zuko.flows import UnconditionalDistribution
from causalflows.flows import CausalFlow
from causal_cocycle.causalflow_helper import select_and_train_flow, sample_do, sample_cf
from causal_cocycle.causalflow_helper import LearnableNormal, LearnableLaplace, LearnableStudentT
from causal_cocycle.helper_functions import ks_statistic, wasserstein1_repeat, rmse

def evaluate_models(
    models_dict: dict,
    index_dict: dict,
    sc_fun: callable,
    X: torch.Tensor,
    Y: torch.Tensor,
    noisedist: Distribution,
    noisetransform: callable,
    seed: int = None,
    N_true: int = 10**5,
    intervention_node: int = 1,
    intervention_fn: callable = torch.full_like,
    intervention_value: float = 0.0,
) -> dict:
    """
    Evaluates each model by comparing its interventional, counterfactual,
    and paired-difference estimates to the ground-truth SCM (`sc_fun`).
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = X.device
    N, d = X.shape
    N, p = Y.shape

    # --- True paired (Y, Y_cf) via SCM generator ---
    torch.manual_seed(seed)
    X_obs_true, U = sc_fun(
        N_true,
        seed=seed,
        intervention_node=None,
        return_u=True,
        noise_dists=[Normal(0,1)] + [noisedist],
        noise_transforms = [lambda x : x] + [noisetransform]
    )
    X_obs_true = X_obs_true.to(device)
    torch.manual_seed(seed)
    X_cf_true, Ucf = sc_fun(
        N_true,
        seed=seed,
        intervention_node=intervention_node,
        intervention_fn=intervention_fn,
       intervention_value=intervention_value,
        return_u=True,
        noise_dists=[Normal(0,1)] + [noisedist],
        noise_transforms = [lambda x : x] + [noisetransform]
    )
    X_cf_true = X_cf_true.to(device)
    # check same noise
    assert((U-Ucf).sum()==0)
    
    # Extract paired Y variables (columns 1 onward)
    Y_true = X_obs_true[:, 1:]
    Y_cf_true = X_cf_true[:, 1:]
    Y_dim = Y_true.shape[1]

    # paired-difference
    diff_true = Y_cf_true - Y_true

    results = {}
    with torch.no_grad():
        for name, (flow, _) in models_dict.items():
            # --- Model paired (Y, Y_cf) via sample_do ---
            torch.manual_seed(seed)
            X_model = sample_do(
                flow.to(device),
                index=intervention_node-1,
                intervention_fn=lambda old: old,
                sample_shape=torch.Size([N_true])
            )
            torch.manual_seed(seed)
            X_cf_model = sample_do(
                flow.to(device),
                index=intervention_node-1,
                intervention_fn=lambda x : intervention_fn(x,intervention_value),
                sample_shape=torch.Size([N_true])
            )
            Y_model = X_model[:, 1:]
            Y_cf_model = X_cf_model[:, 1:]
            diff_model = Y_cf_model - Y_model
    
            # --- Marginal KS values for each Y dimension ---
            w1_vals = [
                wasserstein1_repeat(diff_model[:, j].cpu(), diff_true[:, j].cpu())
                for j in range(Y_dim)
            ]
    
            # --- Marginal KS values for each Y dimension ---
            ks_vals = [
                ks_statistic(diff_model[:, j].cpu(), diff_true[:, j].cpu())
                for j in range(Y_dim)
            ]
    
            # --- Interventional marginal KS ---
            # Reuse Y_cf_model for marginal KS
            w1_int_vals = [
                wasserstein1_repeat(Y_cf_model[:, j].cpu(), Y_cf_true[:, j].cpu())
                for j in range(Y_dim)
            ]
            ks_int_vals = [
                ks_statistic(Y_cf_model[:, j].cpu(), Y_cf_true[:, j].cpu())
                for j in range(Y_dim)
            ]
    
            # --- Counterfactual RMSE ---
            torch.manual_seed(seed)
            X_obs, U = sc_fun(
                N,
                seed=seed,
                intervention_node=None,
                return_u=True,
                noise_dists=[Normal(0,1)] + [noisedist],
                noise_transforms = [lambda x : x] + [noisetransform]
            )
            X_obs = X_obs.to(device)
            Y_obs = X_obs[:, 1:].to(device)
            torch.manual_seed(seed)
            X_cf, Ucf = sc_fun(
                N,
                seed=seed,
                intervention_node=intervention_node,
                intervention_fn=intervention_fn,
                intervention_value=intervention_value,
                return_u=True,
                noise_dists=[Normal(0,1)] + [noisedist],
                noise_transforms = [lambda x : x] + [noisetransform]
            )
            X_cf = X_cf.to(device)
            Y_cf_true = X_cf[:, 1:]

            # Checking consistency of simulated data
            Z = torch.cat([X, Y], dim=1).to(device)
            assert((Z - X_obs).sum() == 0)
            
            Z_cf = sample_cf(
                flow.to(device),
                x_obs=Z,
                index=intervention_node - 1,
                intervention_fn=lambda x : intervention_fn(x,intervention_value),
            )
            rmse_cf_vals = [
                rmse(Z_cf[:, 1:][:,j].cpu(), Y_cf_true[:, j].cpu())
                for j in range(Y_dim)
            ]
            results[name] = {
                'KS_CF': ks_vals,
                'KS_int': ks_int_vals,
                'W1_CF': w1_vals,
                'W1_int': w1_int_vals,
                'RMSE_CF': rmse_cf_vals,
                'index': index_dict[name][0]
            }
    return results

base_map = {
    'normal':   LearnableNormal,
    'laplace':  LearnableLaplace,
    'studentt': LearnableStudentT,
    }

def run_experiment(
    sc_name: str = "chain5_linear",
    noise_dist: str = "normal",
    corr = 0.0,
    seed: int = 0,
    N: int = 1000,
    use_dag = False,
    bases: tuple = ('normal','laplace', 'studentt'),
    num_epochs: int = 1000,
    k_folds: int = 2,
    batch_size: int = 128,
    lr: float = 1e-2,
    learn_flow = True,
) -> dict:
    """
    Runs joint causal-flow experiments for a single noise_dist.
    Returns a dict of results keyed by f"{scm}_{base}".
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Getting SCM
    sc_fun = SCMS[sc_name]
    d = SCM_DIMS[sc_name]
    mask = SCM_MASKS[sc_name] if use_dag else None

    # Getting transformation
    Σ = (1-corr) * torch.eye(d-1) + corr * torch.ones(d-1, d-1)
    L = torch.linalg.cholesky(Σ)

    # set up noise and transforms
    if noise_dist == 'normal':
        noisedist, noisetransform = Normal(0,1), lambda x: x @ L.t()
    elif noise_dist == 'rademacher':
        noisedist, noisetransform = Uniform(-1,1), lambda x: torch.sign(x) @ L.t()
    elif noise_dist == 'cauchy':
        noisedist, noisetransform = Cauchy(0,1), lambda x: x @ L.t()
    elif noise_dist == 'gamma':
        noisedist, noisetransform = Gamma(1,1), lambda x: x @ L.t()
    elif noise_dist == 'inversegamma':
        noisedist, noisetransform = Gamma(1,1), lambda x: 1/x @ L.t()
    else:
        raise ValueError(noise_dist)

    results = {}

    # Intevention_fn
    intervention_fn = torch.full_like #lambda x,a: x+a
    intervention_value = 0.0 # 1.0
    
    # generate data
    V, U = sc_fun(
        N, seed=seed, return_u=True,
        noise_dists=[Normal(0,1)]+[noisedist],
        noise_transforms=[lambda x:x]+[noisetransform]
    )
    V = V.to(device)
    X, Y = V[:,:1], V[:,1:]

    # build candidate flows per base
    transforms = get_stock_coupling_transforms(x_dim=1, y_dim=d-1, mask = mask)
    
    # Lin architecture
    if not learn_flow:
        transforms = transforms[:1]
    
    flow_lrs = [1 * lr] + [1 * lr]* (len(transforms)-1) # smaller lr for NNs 
    for base_type in bases:
        if base_type not in base_map:
            raise ValueError(f"Unknown base type: {base_type}")
        Base = base_map[base_type]
        base = Base(dim=d)
        #base = UnconditionalDistribution(
        #    Base, loc=torch.zeros(d), scale=torch.ones(d), buffer=True
        #)
        flows = [CausalFlow(transform=maf, base=base)
                 for maf in transforms]

        # CV + retrain across all transforms
        best_flow, test_nll, best_idx, cv_scores = select_and_train_flow(
            flows, V, train_fraction=1.0, k_folds=k_folds,
            num_epochs=num_epochs, batch_size=batch_size, lr=lr,
            device=device, flow_lrs = flow_lrs
        )

        key = f"{base_type}"
        
        # evaluate
        with torch.no_grad():
            metrics = evaluate_models(
                {key:(best_flow,'flow')},
                {key:(best_idx,'flow')},
                sc_fun, X, Y,
                noisedist, noisetransform,
                seed=seed,
                intervention_fn = intervention_fn,
                intervention_value = intervention_value,
            )
        
        results[key] = {
            'best_idx': best_idx,
            'cv_scores': cv_scores,
            **metrics
        }

    # add metadata
    results.update({
            'noise':      noise_dist,
            'scm':        sc_name,
        })
    return results


if __name__ == '__main__':
    # Example: run for two different true noise dists
    for nd in ['normal','cauchy']:
        print(f"\n=== RESULTS for true noise = {nd} ===")
        res = run_experiment(noise_dist=nd, seed=0, N=100)
        print(res)

