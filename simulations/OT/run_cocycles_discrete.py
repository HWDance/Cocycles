import torch
import numpy as np
from causal_cocycle.model_new import ZukoCocycleModel
from causal_cocycle.optimise_new import validate
from causal_cocycle.loss import CocycleLoss
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.kernels import gaussian_kernel
from architectures import get_anchored_discrete_flows, get_anchored_discrete_coupling_flow
import os
import math

# === Step 1: Set up SCM ground truth for Y(x) ===

def generate_scm_data(n=250, seed=0, learn_rate = 1e-2, corr = True, affine = True):
    np.random.seed(seed)
    m0 = np.array([0.0, 0.0])
    m1 = np.array([1.0, 1.0])
    m2 = np.array([-1.0, 2.0])

    if corr:
        S0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        S1 = np.array([[1.0, -0.9], [-0.9, 1.0]])
        S2 = np.array([[0.5, 0.5], [0.5, 5.0]])

    else:
        S0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        S1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        S2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        
    L0 = np.linalg.cholesky(S0)
    L1 = np.linalg.cholesky(S1)
    L2 = np.linalg.cholesky(S2)

    xi = np.random.laplace(size = (n, 2))
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
    selectors = get_anchored_discrete_flows(y_dim=y_dim, hidden=(32, 32))
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

def run(seed=0, n=250, learn_rate = 1e-2, wrong_order = False, corr = True, affine = True):
    
    X_raw, Y, xi = generate_scm_data(n=n, seed=seed, corr=corr, affine = affine)
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
    hyper_kwargs = [{'learn_rate': 0.01}] + [{'learn_rate': 0.001}] * (len(models)-1)

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
            "best_idx": best_ind,
            "RMSE10": compute_errors(diff_10, true_diff_10)[0],
            "RMSE21": compute_errors(diff_21, true_diff_21)[0],
            "RMSE20": compute_errors(diff_20, true_diff_20)[0],
            "ATE10": compute_errors(diff_10, true_diff_10)[1],
            "ATE21": compute_errors(diff_21, true_diff_21)[1],
            "ATE20": compute_errors(diff_20, true_diff_20)[1],
        }

        # Define the true L0 (Cholesky of S0 = [[10,0],[0,1]])
        device = next(best_model.parameters()).device

        L0 = torch.tensor([[math.sqrt(10.0), 0.0],
                           [0.0, 1.0]], dtype=torch.float32, device=device)
        
        print("\n=== Recovering DGP’s (m_x, L_x) from the learned A_est, mu_est ===")
        
        for xval in [0, 1, 2]:
            # Build a batch of size 3 with the same xval
            x_batch = torch.full((3, 1), float(xval), dtype=torch.float32, device=device)
        
            # Push the three basis latents through the learned flow: u → y
            u_basis = torch.tensor([
                [0.0, 0.0],    # maps to mu(x)
                [1.0, 0.0],    # maps to mu(x) + A[:,0]
                [0.0, 1.0],    # maps to mu(x) + A[:,1]
            ], dtype=torch.float32, device=device)
        
            y_out = best_model.transformation(x_batch, u_basis)  # shape (3, 2)
        
            # Extract learned mu_est and A_est:
            y0 = y_out[0].view(2, 1)   # mu_est(x)
            y1 = y_out[1].view(2, 1)   # mu_est + A[:,0]
            y2 = y_out[2].view(2, 1)   # mu_est + A[:,1]
        
            col0 = y1 - y0
            col1 = y2 - y0
            A_est = torch.cat([col0, col1], dim=1)   # 2×2 tensor
            mu_est = y0.squeeze(1)                   # length‐2 tensor
        
            # Because we anchored A(0)=I and mu(0)=0, the learned (A_est, mu_est) satisfy:
            #   A_est(x) = L_x @ L0^{-1},   mu_est(x) = m_x - L_x @ L0^{-1} @ m0
            # with m0=0.  Therefore:
            #   L_x = A_est(x) @ L0
            #   m_x = mu_est(x)
        
            L_x = A_est @ L0
            S_x = L_x @ L_x.T
            m_x = mu_est
        
            print(f"\n--- x = {xval} ---")
            print("Learned A_est(x) =")
            print(A_est.detach().cpu().numpy())
            print("Learned mu_est(x) =")
            print(mu_est.detach().cpu().numpy())
            print("→ Recovered true S_x =")
            print(S_x.detach().cpu().numpy())
            print("→ Recovered true m_x =")
            print(m_x.detach().cpu().numpy())
    print("\n===== Cocycle Model Results =====")
    for k, v in results.items():
        print(f"{k:>20}: {v:.6f}")

    return results

if __name__ == "__main__":
    run(n=500, learn_rate = 1e-2)

