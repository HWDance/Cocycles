import numpy as np
import torch

from causal_cocycle.kernels_new import GaussianKernel, median_heuristic

try:
    from .dgp import generate_backdoor_binary_data
    from .helpers import build_KR_map, compute_weights_from_kernel
except ImportError:
    from dgp import generate_backdoor_binary_data
    from helpers import build_KR_map, compute_weights_from_kernel


def _evaluate(est_y0, est_y1, x, y0, y1):
    true_cf = np.where(x == 0, y1, y0)
    est_cf = np.where(x == 0, est_y1, est_y0)
    est_ite = est_y1 - est_y0

    return {
        "RMSE0": np.sqrt(np.mean((est_y0 - y0) ** 2)),
        "RMSE1": np.sqrt(np.mean((est_y1 - y1) ** 2)),
        "RMSEcf": np.sqrt(np.mean((est_cf - true_cf) ** 2)),
        "PEHE": np.sqrt(np.mean((est_ite - 1.0) ** 2)),
        "ATE": est_ite.mean(),
        "ATEError": abs(est_ite.mean() - 1.0),
    }


def _conditional_kr_counterfactuals(y, c, x, epsilon=0.0):
    y_tensor = torch.tensor(y, dtype=torch.float64)
    c_tensor = torch.tensor(c, dtype=torch.float64)
    if c_tensor.dim() == 1:
        c_tensor = c_tensor.unsqueeze(-1)

    mask0 = x == 0
    mask1 = x == 1

    y0_obs = y_tensor[mask0]
    y1_obs = y_tensor[mask1]
    c0 = c_tensor[mask0]
    c1 = c_tensor[mask1]

    ls = median_heuristic(c_tensor)
    if torch.isnan(ls) or ls.item() == 0:
        ls = torch.tensor(1.0, dtype=torch.float64)
    kernel = GaussianKernel(lengthscale=ls.clone().detach().to(dtype=torch.float64))

    est_y0 = y.copy()
    est_y1 = y.copy()

    for i in np.where(mask0)[0]:
        c_i = c_tensor[i]
        w_src = compute_weights_from_kernel(kernel, c0, c_i)
        w_tgt = compute_weights_from_kernel(kernel, c1, c_i)
        kr = build_KR_map(y0_obs, y1_obs, w_src=w_src, w_tgt=w_tgt, epsilon=epsilon)
        est_y1[i] = kr(y_tensor[i]).item()

    for i in np.where(mask1)[0]:
        c_i = c_tensor[i]
        w_src = compute_weights_from_kernel(kernel, c1, c_i)
        w_tgt = compute_weights_from_kernel(kernel, c0, c_i)
        kr = build_KR_map(y1_obs, y0_obs, w_src=w_src, w_tgt=w_tgt, epsilon=epsilon)
        est_y0[i] = kr(y_tensor[i]).item()

    return est_y0, est_y1


def run(
    seed=0,
    n=500,
    rho=0.5,
    logitscale=1.0,
    condition_on_z=False,
    epsilon=0.0,
    use_c=None,
):
    if use_c is not None:
        condition_on_z = use_c

    data = generate_backdoor_binary_data(seed=seed, n=n, rho=rho, logitscale=logitscale, y_dim=1)

    x = data["X"]
    z = data["Z_obs"].numpy()
    y = data["Y"]
    y0 = np.asarray(data["Y0"]).reshape(-1)
    y1 = np.asarray(data["Y1"]).reshape(-1)

    mask0 = x == 0
    mask1 = x == 1
    if mask0.sum() == 0 or mask1.sum() == 0:
        raise ValueError("Back-door seqOT requires both treatment arms to be present.")

    if condition_on_z:
        est_y0, est_y1 = _conditional_kr_counterfactuals(y, z, x, epsilon=epsilon)
    else:
        y0_obs = torch.tensor(y[mask0], dtype=torch.float64)
        y1_obs = torch.tensor(y[mask1], dtype=torch.float64)
        kr01 = build_KR_map(y0_obs, y1_obs, epsilon=epsilon)
        kr10 = build_KR_map(y1_obs, y0_obs, epsilon=epsilon)

        est_y0 = y.copy()
        est_y1 = y.copy()
        est_y1[mask0] = kr01(y0_obs).numpy()
        est_y0[mask1] = kr10(y1_obs).numpy()

    results = {
        "seed": seed,
        "name": "backdoor_seqot_z" if condition_on_z else "backdoor_seqot",
        "rho": rho,
        "logitscale": logitscale,
        "treated_share": float(np.mean(x)),
        "condition_on_z": condition_on_z,
        "epsilon": epsilon,
    }
    results.update(_evaluate(est_y0, est_y1, x, y0, y1))
    return results


if __name__ == "__main__":
    result = run(seed=0, n=200, rho=0.5, logitscale=1.0, condition_on_z=True, epsilon=0.0)
    print("\n===== Backdoor seqOT Smoke Test =====")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:>20}: {v:.6f}")
        else:
            print(f"{k:>20}: {v}")
