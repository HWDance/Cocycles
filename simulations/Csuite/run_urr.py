import torch
import torch.nn as nn
from torch.distributions import (
    Distribution, Normal, Laplace, Cauchy, Gamma, Uniform
)
# Csuite Imports
from csuite import SCMS, SCM_DIMS, SCM_MASKS
from architectures import get_stock_transforms
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.optimise_new import validate
from causal_cocycle.kernels import gaussian_kernel
from causal_cocycle.model_new import ZukoFlowModel
from causal_cocycle.transformers_new import Transformer, ShiftLayer
from causal_cocycle.helper_functions import ks_statistic, wasserstein1_repeat, rmse
from causal_cocycle.causalflow_helper import LearnableNormal, LearnableLaplace, LearnableStudentT
import random


import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, Laplace, StudentT, Independent


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
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    device = X.device
    N, d = X.shape
    _, p = Y.shape

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
    assert((U-Ucf).sum()==0)

    Y_true = X_obs_true[:, 1:]
    Y_cf_true = X_cf_true[:, 1:]
    Y_dim = Y_true.shape[1]
    diff_true = Y_cf_true - Y_true

    results = {}
    with torch.no_grad():
        for name, (model, _) in models_dict.items():
            Y_model = model.cocycle(X, X, Y)
            Y_cf_model = model.cocycle(intervention_fn(X, intervention_value), X, Y)
            diff_model = Y_cf_model - Y_model

            w1_vals = [
                wasserstein1_repeat(diff_model[:, j].cpu(), diff_true[:, j].cpu())
                for j in range(Y_dim)
            ]
            ks_vals = [
                ks_statistic(diff_model[:, j].cpu(), diff_true[:, j].cpu())
                for j in range(Y_dim)
            ]

            X_sub = X[torch.randint(0, len(X), (N_true,))].to(device)
            X_cf_sub = intervention_fn(X_sub, intervention_value)
            base_samples = model.base_distribution.sample((N_true, )).to(device)
            Y_cf_gen = model.transformation(X_cf_sub, base_samples)

            w1_int_vals = [
                wasserstein1_repeat(Y_cf_gen[:, j].cpu(), Y_cf_true[:, j].cpu())
                for j in range(Y_dim)
            ]
            ks_int_vals = [
                ks_statistic(Y_cf_gen[:, j].cpu(), Y_cf_true[:, j].cpu())
                for j in range(Y_dim)
            ]

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
    sc_name: str = "2var_linear",
    noise_dist: str = "normal",
    corr = 0.0,
    seed: int = 0,
    N: int = 1000,
    use_dag = False,
    num_epochs: int = 1000,
    k_folds: int = 2,
    batch_size: int = 128,
    lr: float = 1e-2,
    learn_flow = True,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sc_fun = SCMS[sc_name]
    d = SCM_DIMS[sc_name]
    mask = SCM_MASKS[sc_name][1:][:,1:] if (use_dag and d > 2) else None

    Σ = (1 - corr) * torch.eye(d - 1) + corr * torch.ones(d - 1, d - 1)
    L = torch.linalg.cholesky(Σ)

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

    V, U = sc_fun(
        N, seed=seed, return_u=True,
        noise_dists=[Normal(0,1)] + [noisedist],
        noise_transforms=[lambda x: x] + [noisetransform]
    )
    V = V.to(device)
    X, Y = V[:, :1], V[:, 1:]

    # Intevention_fn
    intervention_fn = torch.full_like #lambda x,a: x+a
    intervention_value = 0.0 # 1.0

    transforms = get_stock_transforms(x_dim=1, y_dim=d-1, mask=mask)
    kernel = [gaussian_kernel()] * 2
    loss_factory = CocycleLossFactory(kernel)

    metrics = {}
    #for base_name, base_dist in zip(['G', 'L'], [Normal(torch.zeros(d-1), torch.ones(d-1)),
    #                                             Laplace(torch.zeros(d-1), torch.ones(d-1))]):
    for base_name, base_dist in zip(['G', 'L', 'T'], [LearnableNormal(d-1),
                                                     LearnableLaplace(d-1),
                                                      LearnableStudentT(d-1)]):
        models = [
            ZukoFlowModel(nn.ModuleList(t), base_dist) for t in transforms
        ]
        model_names = [f"URR_{base_name}_{i}" for i in range(len(models))]

        urr = loss_factory.build_loss("URR", X, Y)
        urr_val = loss_factory.build_loss("URR_N", X, Y)

        opt_config = {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "val_batch_size": 1024,
            "learn_rate": lr,
            "print_": False,
        }

        final_model, (idx, _) = validate(
            models, urr, X, Y, method="CV", loss_val = urr_val,
            train_val_split=0.5, opt_kwargs=opt_config,
            hyper_kwargs=[{"learn_rate": lr}] * len(models),
            choose_best_model="overall",
            retrain=True
        )

        model_key = f"URR_{base_name}"
        model_dict = {model_key: (final_model, model_key)}
        index_dict = {model_key: (idx, model_key)}

        base_metrics = evaluate_models(model_dict, index_dict, sc_fun, X, Y,
                                       noisedist, noisetransform, seed=seed,
                                       intervention_fn=intervention_fn,
                                       intervention_value=intervention_value)
        metrics.update(base_metrics)

    metrics.update({'noise': noise_dist, 'scm': sc_name})
    return metrics


if __name__ == '__main__':
    results = run_experiment(seed=0)
    print(results)
