import numpy as np
import torch
from causal_cocycle.kernels_new import GaussianKernel, median_heuristic, median_heuristic_ard
from helpers import multivariate_laplace, compute_weights_from_kernel, build_KR_map

# ----------------------------
# End-to-end KR transport procedure
# ----------------------------

def run(seed, n = None, m = None, epsilon = 0, wrongorder = False, additive = True, corr = 0.5, multivariate_noise = False, dist = "laplace"):

    np.random.seed(seed)
    n0 = n1 = n2 = n or 250
    if m is not None:
        n1 = m

    # Sample from the DGP
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
            xi0,xi1,xi2 = (
                multivariate_laplace(size = n0, rng = seed, corr = corr),
                multivariate_laplace(size = n1, rng = seed+1, corr = corr),
                multivariate_laplace(size = n2, rng = seed+2, corr = corr)
        )
        else:
            cov = np.ones((2,2))*corr + (1-corr)*np.eye(2)
            xi0,xi1,xi2 = (
                np.random.multivariate_normal(size = n0, mean = np.zeros(2), cov = cov),
                np.random.multivariate_normal(size = n0, mean = np.zeros(2), cov = cov),
                np.random.multivariate_normal(size = n0, mean = np.zeros(2), cov = cov),
        )                
        
        xi0[:,1] = xi0[:,0] + xi0[:,1]
        xi1[:,1] = xi1[:,0] + xi1[:,1]
        xi2[:,1] = xi2[:,0] + xi2[:,1]
    else:
        if dist == "laplace":
            xi0,xi1,xi2 = ( 
                np.random.laplace(size = (n0,2)),
                np.random.laplace(size = (n1,2)),
                np.random.laplace(size = (n2,2))
            )
        else:
            xi0,xi1,xi2 = ( 
                np.random.normal(size = (n0,2)),
                np.random.normal(size = (n1,2)),
                np.random.normal(size = (n2,2))
            )
    P0 = m0 + xi0 @ L0.T
    P1 = m1 + xi1 @ L1.T
    P2 = m2 + xi2 @ L2.T

    if not affine:
        P1 = 1/(1+np.exp(-P1))*5
        P2 = 1/(1+np.exp(-P2))*10

    
    # Convert to torch tensors
    Y0 = torch.tensor(P0, dtype=torch.float64)
    Y1 = torch.tensor(P1, dtype=torch.float64)
    Y2 = torch.tensor(P2, dtype=torch.float64)

    
    # wrong order adjustment
    if wrongorder:
        Y0 = Y0.flip(dims=[1])
        Y1 = Y1.flip(dims=[1])
        Y2 = Y2.flip(dims=[1])

    # Merging for sparsification 
    Y = torch.column_stack((Y0,Y1,Y2))
    
    # ----------------------------
    # KR transport for the first dimension (1D maps)
    # Use uniform weights here (empirical CDF)
    KR10 = build_KR_map(Y0[:,0], Y1[:,0], epsilon=epsilon)
    KR21 = build_KR_map(Y1[:,0], Y2[:,0], epsilon=epsilon)
    KR20 = build_KR_map(Y0[:,0], Y2[:,0], epsilon=epsilon)
    
    Yhat1 = KR10(Y0[:,0])           # Transport P0 -> P1, first coordinate
    Yhat2_direct = KR20(Y0[:,0])      # Direct transport P0 -> P2, first coordinate
    Yhat2_composite = KR21(Yhat1)     # Composite transport P0 -> P1 -> P2, first coordinate
    
    # ----------------------------
    # For the second dimension we use the weighted KR transport.
    ls_best = median_heuristic(Y[:,:1])
    kernel = GaussianKernel(lengthscale=torch.tensor(ls_best))
    
    mapped_second_direct = []
    mapped_second_composite = []
    mapped_second_1 = []
    
    # For each sample in P0, compute the conditional KR transport for the second dimension.
    for i in range(Y0.shape[0]):
        # Extract source coordinates from P0:
        x1_0 = Y0[i, 0]
        x2_0 = Y0[i, 1]
        X2 = Y[:, 1]  # second coordinates of P0
        
        # Compute source conditional weights from the first coordinate using uniform (empirical) weighting
        # For the conditional KR on the second dimension for P0 -> P1:
        w_src = compute_weights_from_kernel(kernel, Y[:,0], x1_0)
        
        # ----- For P0 -> P1 (for the second coordinate) -----
        # Compute target conditional weights using the first-coordinate mapping from P0->P1
        w_tgt_1 = compute_weights_from_kernel(kernel, Y[:,0], KR10(x1_0))
        Y1_2 = Y[:,1]
        KR_2_1 = build_KR_map(X2, Y1_2, w_src, w_tgt_1, epsilon)
        mapped_second_1.append(KR_2_1(x2_0).item())
        
        # ----- For direct P0 -> P2 -----
        w_tgt_direct = compute_weights_from_kernel(kernel, Y[:,0], KR20(x1_0))
        Y2_direct = Y[:,1]
        KR_2_direct = build_KR_map(X2, Y2_direct, w_src, w_tgt_direct, epsilon)
        mapped_second_direct.append(KR_2_direct(x2_0).item())
        
        # ----- For composite transport P0 -> P1 -> P2 -----
        # Use the first-dim composite mapping: x1_hat from Yhat1 (which is P0 -> P1)
        x1_hat = Yhat1[i]
        w_tgt_composite = compute_weights_from_kernel(kernel, Y[:,0], KR21(x1_hat))
        KR_2_composite = build_KR_map(X2, Y[:,1], w_src, w_tgt_composite, epsilon)
        mapped_second_composite.append(KR_2_composite(x2_0).item())
    
    mapped_second_1 = torch.tensor(mapped_second_1)
    mapped_second_direct = torch.tensor(mapped_second_direct)
    mapped_second_composite = torch.tensor(mapped_second_composite)
    
    # Construct full 2D transported points:
    Yhat1_2d = torch.stack([Yhat1, mapped_second_1], dim=1)
    Yhat2_direct_2d = torch.stack([Yhat2_direct, mapped_second_direct], dim=1)
    Yhat2_composite_2d = torch.stack([Yhat2_composite, mapped_second_composite], dim=1)

    # ------------------------------
    # Ground truth from SCM model

    # Recover latent noise ξ from P0
    L0inv = np.linalg.inv(L0)
    xi_hat = (P0 - m0) @ L0inv.T  # each row is a ξ sample

    # Compute counterfactual outcomes using shared ξ
    Y1_cf = m1 + xi_hat @ L1.T
    Y2_cf = m2 + xi_hat @ L2.T

    # True counterfactual shifts
    true_shift_10 = Y1_cf - P0
    true_shift_21 = Y2_cf - Y1_cf
    true_shift_20 = Y2_cf - P0
    
    # ----------------------------
    # Compute error statistics relative to the true shifts
    
    # Compute error for P0 -> P1:
    # Estimated shift for P0 -> P1 (full 2D) is Yhat1_2d - Y0.
    delta_10 = (Yhat1_2d - Y0).numpy()     # shape: (n, 2)
    error_10 = delta_10 - true_shift_10      # error compared to [3, 3]
    norm_error_10 = np.linalg.norm(error_10, axis=1)  # L2 norm per sample
    
    # Compute error for P1 -> P2:
    # For the direct KR transport, the estimated shift is (Yhat2_direct_2d - Yhat1_2d)
    delta_21_direct = (Yhat2_direct_2d - Yhat1_2d).numpy()
    error_21_direct = delta_21_direct - true_shift_21
    norm_error_21_direct = np.linalg.norm(error_21_direct, axis=1)
    
    # For the composite KR transport, the estimated shift is (Yhat2_composite_2d - Yhat1_2d)
    delta_21_composite = (Yhat2_composite_2d - Yhat1_2d).numpy()
    error_21_composite = delta_21_composite - true_shift_21
    norm_error_21_composite = np.linalg.norm(error_21_composite, axis=1)
    
    # Compute error for P1 -> P2:
    # For the direct KR transport, the estimated shift is (Yhat2_direct_2d - Yhat1_2d)
    delta_20_direct = (Yhat2_direct_2d - Y0).numpy()
    error_20_direct = delta_20_direct - true_shift_20
    norm_error_20_direct = np.linalg.norm(error_20_direct, axis=1)
    
    # For the composite KR transport, the estimated shift is (Yhat2_composite_2d - Yhat1_2d)
    delta_20_composite = (Yhat2_composite_2d - Y0).numpy()
    error_20_composite = delta_20_composite - true_shift_20
    norm_error_20_composite = np.linalg.norm(error_20_composite, axis=1)

    # Compute inconsistency
    norm_inconsistency = np.linalg.norm(Yhat2_direct_2d-Yhat2_composite_2d, axis = 1)
    
    obj = {
        "seed": seed,
        "name": "KReps{0}".format(epsilon),
        "corr": corr,
        "additive": additive,
        "wrongorder" : wrongorder,
        "RMSE10" : norm_error_10.mean(),
        "RMSE21direct" : norm_error_21_direct.mean(),
        "RMSE21composite" : norm_error_21_composite.mean(),
        "RMSE20direct" : norm_error_20_direct.mean(),
        "RMSE20composite" : norm_error_20_composite.mean(),
        "RMSEinconsistency" : norm_inconsistency.mean(),
        "ATE10" :  ((error_10.mean(0))**2).mean()**0.5,
        "ATE21direct" : ((error_21_direct.mean(0))**2).mean()**0.5,
        "ATE21composite" :  ((error_21_composite.mean(0))**2).mean()**0.5,
        "ATE20direct" :  ((error_20_direct.mean(0))**2).mean()**0.5,
        "ATE20composite" :  ((error_20_composite.mean(0))**2).mean()**0.5,
    }
    
    return obj