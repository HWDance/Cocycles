import torch
import numpy as np
from causal_cocycle.model_new import ZukoCocycleModel
from causal_cocycle.optimise_new import validate
from causal_cocycle.loss import CocycleLoss
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.kernels import gaussian_kernel
from architectures import get_anchored_discrete_flows, get_anchored_discrete_flows_single
import os
import math

def multivariate_laplace(mu = np.zeros(2), corr = 0.25, b=1.0, size=1, rng=None):
    """
    Sample from a multivariate Laplace distribution using the
    Gaussian-exponential mixture representation.

    Parameters
    ----------
    mu : array_like, shape (d,)
        Mean vector.
    Sigma : array_like, shape (d, d)
        Covariance matrix (positive definite).
    b : float
        Scale parameter (Laplace 'spread').
    size : int
        Number of samples.
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    samples : ndarray, shape (size, d)
        Multivariate Laplace samples.
    """
    rng = np.random.default_rng(rng)
    mu = np.asarray(mu)
    d = mu.shape[0]

    # Cholesky of covariance
    Sigma = np.ones((d,d))*corr + (1-corr)*np.eye(d)
    L = np.linalg.cholesky(Sigma)

    # Exponential mixture variable
    W = rng.exponential(scale=b, size=size)

    # Multivariate normal samples
    Z = rng.normal(size=(size, d)) @ L.T

    # Combine
    return mu + np.sqrt(W)[:, None] * Z

# === Step 1: Set up SCM ground truth for Y(x) ===

def generate_scm_data(n=250, seed=0, learn_rate = 1e-2, corr = 0.5, affine = True, additive = True, multivariate_noise = False, dist = "laplace"):
    np.random.seed(seed)
    m0 = np.array([0.0, 0.0])
    m1 = np.array([1.0, 1.0])
    m2 = np.array([2.0, 2.0])

    if not additive:
        S0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        S1 = np.array([[1.0, -corr], [-corr, 1.0]])
        S2 = np.array([[(1+corr), 0.0], [0.0, (1/(1+corr))]])

    else:
        S0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        S1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        S2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        
    L0 = np.linalg.cholesky(S0)
    L1 = np.linalg.cholesky(S1)
    L2 = np.linalg.cholesky(S2)

    if multivariate_noise:
        if dist == "laplace":
            xi= multivariate_laplace(size = n, rng = seed, corr = corr)
        else:
            cov = np.ones((2,2))*corr + (1-corr)*np.eye(2)
            xi = np.random.multivariate_normal(size = n, mean = torch.zeros(2), cov = cov),
        xi[:,1] = xi[:,0] + xi[:,1]
    else:
        if dist == "laplace":
            xi = np.random.laplace(size = (n,2))
        else:
            xi = np.random.normal(size = (n,2))
            
    Y0 = m0 + xi @ L0.T
    Y1 = m1 + xi @ L1.T
    Y2 = m2 + xi @ L2.T

    if not affine:
        Y1 = 1/(1+np.exp(-Y1))*5
        Y2 = 1/(1+np.exp(-Y2))*10

    X = np.concatenate([
        np.tile([0], n),
        np.tile([1], n),
        np.tile([2], n)
    ])[:, None]

    Y = np.concatenate([Y0, Y1, Y2])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), xi

# === Step 2: Build ZukoCocycleModels ===

def build_models(x_dim, y_dim):
    models = []
    # Create all four anchored‐discrete architectures
    selectors = get_anchored_discrete_flows_single(y_dim=y_dim, hidden=(32, 32))
    # For each selector transform, wrap it in a ModuleList and build a ZukoCocycleModel
    for selector in selectors:
        models.append(
            ZukoCocycleModel(transforms=torch.nn.ModuleList([selector]))
        )
    return models

# === Step 3: Loss and kernel ===

def build_loss(X, Y):
    kernel = [gaussian_kernel(), gaussian_kernel()]
    loss_factory = CocycleLossFactory(kernel)
    return loss_factory.build_loss("CMMD_U", X, Y)

# === Step 4: Run validation + evaluation ===

def run(seed=0, n=250, learn_rate = 1e-3, wrong_order = False, corr = 0.5, affine = True, additive = True, multivariate_noise = False, dist = "laplace"):
    
    X_raw, Y, xi = generate_scm_data(n=n, seed=seed, corr=corr, affine = affine, additive = additive, multivariate_noise = multivariate_noise, dist = dist)
    if wrong_order:
        Y = Y.flip(dims=[1])
    x_dim = 1
    y_dim = Y.shape[1]
    # Shuffle data
    perm = torch.randperm(X_raw.size(0))
    X_tr = X_raw[perm]
    Y_tr = Y[perm]

    opt_kwargs = dict(
        epochs=1000,
        learn_rate=learn_rate,
        scheduler=False,
        batch_size=128,
        print_=True,
    )

    # Fit cocycle model
    models = build_models(x_dim, y_dim)
    loss = build_loss(X_tr, Y_tr)
    hyper_kwargs = [{'learn_rate': learn_rate}] * (len(models))

    best_model, (best_ind, _) = validate(
        models, loss, X_tr, Y_tr,
        method="fixed",
        train_val_split=0.5,
        choose_best_model="overall",
        retrain=True,
        opt_kwargs=opt_kwargs,
        hyper_kwargs=hyper_kwargs,
    )

    # Estimate counterfactual outcomes using best model
    with torch.no_grad():
        P0 = Y[:n]
        x0 = torch.zeros((n, 1))
        x1 = torch.ones((n, 1))
        x2 = 2 * torch.ones((n, 1))

        Y1_cf = best_model.cocycle(x1, x0, P0)
        Y2_cf = best_model.cocycle(x2, x0, P0)

        true_Y1_cf = Y[n:2*n]
        true_Y2_cf = Y[2*n:3*n]

        diff_10 = Y1_cf - P0
        diff_21 = Y2_cf - Y1_cf
        diff_20 = Y2_cf - P0

        true_diff_10 = true_Y1_cf - P0
        true_diff_21 = true_Y2_cf - true_Y1_cf
        true_diff_20 = true_Y2_cf - P0

        def compute_errors(est, true):
            err = est - true
            rmse = torch.norm(err, dim=1).mean().item()
            ate = torch.sqrt((err.mean(0)**2).mean()).item()
            return rmse, ate

        results = {
            "seed" : seed,
            "name" : "MAFcocycle",
            "corr": corr,
            "additive" : additive,
            "best_idx": best_ind,
            "RMSE10": compute_errors(diff_10, true_diff_10)[0],
            "RMSE21": compute_errors(diff_21, true_diff_21)[0],
            "RMSE20": compute_errors(diff_20, true_diff_20)[0],
            "ATE10": compute_errors(diff_10, true_diff_10)[1],
            "ATE21": compute_errors(diff_21, true_diff_21)[1],
            "ATE20": compute_errors(diff_20, true_diff_20)[1],
        }

    return results

if __name__ == "__main__":
    run(n=500, learn_rate = 1e-2)

