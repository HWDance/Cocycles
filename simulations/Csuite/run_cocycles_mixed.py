import torch
import torch.nn as nn
import torch
from torch.distributions import (
    Distribution, Normal, Laplace, Cauchy, Gamma, Uniform
)
# Csuite Imports
from csuite_mixed import SCMS, SCM_DIMS, SCM_MASKS
from architectures import get_stock_transforms
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.optimise_new import validate
from causal_cocycle.kernels import gaussian_kernel
from causal_cocycle.model_new import ZukoCocycleModel
from causal_cocycle.helper_functions import ks_statistic, wasserstein1_repeat, rmse


def evaluate_models(
    models_dict: dict,
    index_dict: dict,
    sc_fun: callable,
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = None,
    N_true: int = 10**5,
    intervention_node: int = 1,
    intervention_value: float = 0.0,
) -> dict:
    """
    Evaluates each CocycleModel on interventional, counterfactual, and paired-difference metrics.
    Uses ground-truth SCM (`sc_fun`) for reference.
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = X.device
    N, d = X.shape
    _, p = Y.shape

    # --- True paired (Y, Y_cf) via SCM generator ---
    torch.manual_seed(seed)
    X_obs_true, U = sc_fun(
        N_true,
        seed=seed,
        intervention_node=None,
        return_u=True,
    )
    X_obs_true = X_obs_true.to(device)
    torch.manual_seed(seed)
    X_cf_true, Ucf = sc_fun(
        N_true,
        seed=seed,
        intervention_node=intervention_node,
        intervention_value=intervention_value,
        return_u=True,
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
        for name, (model, _) in models_dict.items():
    
            # Counterfactuals with cocycle(X', X, Y)
            Y_model = model.cocycle(X, X, Y)
            Y_cf_model = model.cocycle(X*0 + intervention_value, X, Y)
            diff_model = Y_cf_model - Y_model  # shape (N, p)
            
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
            w1_int_vals = [
                wasserstein1_repeat(Y_cf_model[:, j].cpu(), Y_cf_true[:, j].cpu())
                for j in range(Y_dim)
            ]
            ks_int_vals = [
                ks_statistic(Y_cf_model[:, j].cpu(), Y_cf_true[:, j].cpu())
                for j in range(Y_dim)
            ]
    
            # ---- Counterfactual RMSE using cocycle model ----
            torch.manual_seed(seed)
            X_obs, U = sc_fun(
                N,
                seed=seed,
                intervention_node=None,
                return_u=True,
            )
            X_obs = X_obs.to(device)
            Y_obs = X_obs[:, 1:].to(device)
            
            torch.manual_seed(seed)
            X_cf, Ucf = sc_fun(
                N,
                seed=seed,
                intervention_node=intervention_node,
                intervention_value=intervention_value,
                return_u=True,
            )
            X_cf = X_cf.to(device)
            Y_cf_true = X_cf[:, 1:]
            
            # Consistency check
            Z = torch.cat([X, Y], dim=1).to(device)
            assert((Z - X_obs).sum() == 0)
                
            rmse_cf_vals = [
                rmse(Y_cf_model[:, j].cpu(), Y_cf_true[:, j].cpu())
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

def run_experiment(
    sc_name: str = "chain5_linear",
    corr = 0.0,
    seed: int = 0,
    N: int = 1000,
    use_dag = False,
    num_epochs: int = 1000,
    k_folds: int = 2,
    batch_size: int = 128,
    lr: float = 1e-2,
) -> dict:
    """
    Runs experiments across all SCMs for a given noise distribution, seed, and sample size N.

    Returns a dict mapping sc_name to its metrics.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cocycle opt configs
    opt_config = {
    "epochs": num_epochs,            # Number of epochs per optimisation run.
    "batch_size": batch_size,         # Training batch size.
    "val_batch_size": 1024,         # Training batch size.
    "learn_rate": lr,       # Base learning rate for model parameters.
    "print_": False          # AMEND THIS
}

    # Getting SCM
    sc_fun = SCMS[sc_name]
    d = SCM_DIMS[sc_name]
    mask = SCM_MASKS[sc_name][1:][:,1:] if (use_dag and d > 2) else None
    
    # generate data
    V, U = sc_fun(
        N, seed=seed, return_u=True,
    )
    V = V.to(device)
    X, Y = V[:,:1], V[:,1:]

    # Build architectures
    transforms = get_stock_transforms(x_dim = 1, y_dim = d-1, mask = mask)

    # Cocycle models
    cocycle_models = [ZukoCocycleModel(nn.ModuleList(t)) for t in transforms]

    # Set up loss factories
    kernel = [gaussian_kernel()] * 2
    loss_factory = CocycleLossFactory(kernel)

    # Cross-validate each family
    # Cocycle CMMD (flow-specific LR's_
    hyper_kwargs = [{'learn_rate': lr}] + [{'learn_rate': lr * 1}] * (len(cocycle_models)-1)
    cmmdv = loss_factory.build_loss("CMMD_V", X, Y)
    final_v, (idx_v, _) = validate(cocycle_models, cmmdv, X, Y, method="CV",
                                   train_val_split=0.5, opt_kwargs=opt_config,
                                   hyper_kwargs=hyper_kwargs, choose_best_model="overall", retrain=True)

    # Collect models & indices
    models = {
        'Cocycle_CMMD_V': (final_v, 'cmmdv'),
    }
    idxs = {
        'Cocycle_CMMD_V': (idx_v, 'cmmdv'),
    }

    metrics = evaluate_models(models, idxs, sc_fun, X, Y, seed=seed)
    metrics.update({'scm': sc_name})

    return metrics


if __name__ == '__main__':
    results = run_experiment()
    print(results)
