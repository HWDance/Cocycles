import torch

from causal_cocycle.kernels import gaussian_kernel
from causal_cocycle.kernels_new import GaussianKernel, median_heuristic
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.model_new import ZukoCocycleModel
from causal_cocycle.optimise_new import validate
from causal_cocycle.rkhs_functionals import KRRFunctional
from causal_cocycle.rkhs_weight_estimation import RKHSWeightEstimator

try:
    from .architectures import get_anchored_discrete_flows_single
    from .dgp import generate_backdoor_binary_data
except ImportError:
    from architectures import get_anchored_discrete_flows_single
    from dgp import generate_backdoor_binary_data


def _evaluate(est_y0, est_y1, x, y0, y1):
    true_cf = torch.where(x == 0, y1, y0)
    est_cf = torch.where(x == 0, est_y1, est_y0)
    est_ite = est_y1 - est_y0

    return {
        "RMSE0": torch.sqrt(torch.mean((est_y0 - y0) ** 2)).item(),
        "RMSE1": torch.sqrt(torch.mean((est_y1 - y1) ** 2)).item(),
        "RMSEcf": torch.sqrt(torch.mean((est_cf - true_cf) ** 2)).item(),
        "PEHE": torch.sqrt(torch.mean((est_ite - 1.0) ** 2)).item(),
        "ATE": est_ite.mean().item(),
        "ATEError": abs(est_ite.mean().item() - 1.0),
    }


def build_models(x_dim, y_dim):
    models = []
    selectors = get_anchored_discrete_flows_single(y_dim=y_dim, hidden=(32, 32))
    for selector in selectors:
        models.append(
            ZukoCocycleModel(transforms=torch.nn.ModuleList([selector]))
        )
    return models


def _build_weight_estimator(X, C, Y, tune=True):
    x_dim = X.shape[1]
    c_dim = C.shape[1]

    kernel_u = GaussianKernel(lengthscale=torch.ones(x_dim + c_dim))
    y_lengthscale = median_heuristic(Y)
    if torch.isnan(y_lengthscale) or y_lengthscale.item() == 0:
        y_lengthscale = torch.tensor(1.0, dtype=Y.dtype)
    kernel_y = GaussianKernel(lengthscale=y_lengthscale.reshape(1).to(dtype=Y.dtype))
    functional = KRRFunctional(kernel_u, penalty=1e-3)
    estimator = RKHSWeightEstimator(functional, kernel_y=kernel_y)
    if tune:
        estimator.tune(
            X,
            C,
            Y,
            maxiter=25,
            nfold=3,
            learn_rate=5e-2,
            print_=False,
        )
    estimator.fit(X, C, Y)
    return estimator


def build_loss(inputs, outputs, estimator):
    kernel = [gaussian_kernel(), gaussian_kernel()]
    loss_factory = CocycleLossFactory(kernel)
    return loss_factory.build_loss(
        "WCMMD_V",
        X=inputs,
        Y=outputs,
        weight_estimator=estimator,
        weight_mode="fixed",
    )


def run(
    seed=0,
    n=500,
    rho=0.5,
    logitscale=1.0,
    y_dim=1,
    learn_rate=1e-2,
    epochs=500,
    batch_size=128,
    print_=True,
):
    data = generate_backdoor_binary_data(seed=seed, n=n, rho=rho, logitscale=logitscale, y_dim=y_dim)

    X_raw = data["X_obs"]
    C_raw = data["Z_obs"]
    Y = data["Y_obs"]
    idx = torch.arange(len(X_raw))

    # Shuffle data exactly as in the existing OT cocycle runner.
    perm = torch.randperm(X_raw.size(0))
    X_tr = X_raw[perm]
    C_tr = C_raw[perm]
    Y_tr = Y[perm]
    idx_tr = idx[perm]

    inputs_tr = {
        "X": X_tr,
        "C": C_tr,
        "__idx__": idx_tr,
    }

    estimator = _build_weight_estimator(X_raw, C_raw, Y, tune=True)
    loss = build_loss(inputs_tr, Y_tr, estimator)

    x_dim = 1
    out_dim = Y.shape[1]
    models = build_models(x_dim, out_dim)
    hyper_kwargs = [{"learn_rate": learn_rate}] * len(models)
    opt_kwargs = dict(
        epochs=epochs,
        learn_rate=learn_rate,
        scheduler=False,
        batch_size=min(batch_size, n),
        print_=print_,
    )

    best_model, (best_ind, _) = validate(
        models,
        loss,
        inputs_tr,
        Y_tr,
        method="fixed",
        train_val_split=0.5,
        choose_best_model="overall",
        retrain=True,
        opt_kwargs=opt_kwargs,
        hyper_kwargs=hyper_kwargs,
    )

    x = X_raw.squeeze(-1)
    y0 = torch.tensor(data["Y0"], dtype=torch.float32)
    y1 = torch.tensor(data["Y1"], dtype=torch.float32)

    with torch.no_grad():
        x0 = torch.zeros_like(X_raw)
        x1 = torch.ones_like(X_raw)
        est_y0 = best_model.cocycle(x0, X_raw, Y)
        est_y1 = best_model.cocycle(x1, X_raw, Y)

        mask0 = x == 0
        mask1 = x == 1
        est_y0[mask0] = Y[mask0]
        est_y1[mask1] = Y[mask1]

    if est_y0.dim() > 1 and est_y0.shape[1] > 1:
        est_y0_eval = est_y0[:, 0]
        est_y1_eval = est_y1[:, 0]
        y0_eval = y0[:, 0]
        y1_eval = y1[:, 0]
    else:
        est_y0_eval = est_y0.squeeze(-1)
        est_y1_eval = est_y1.squeeze(-1)
        y0_eval = y0.squeeze(-1)
        y1_eval = y1.squeeze(-1)

    results = {
        "seed": seed,
        "name": "backdoor_cocycle",
        "rho": rho,
        "logitscale": logitscale,
        "y_dim": y_dim,
        "treated_share": x.float().mean().item(),
        "best_idx": best_ind,
    }
    results.update(_evaluate(est_y0_eval, est_y1_eval, x, y0_eval, y1_eval))
    return results


if __name__ == "__main__":
    result = run(
        seed=0,
        n=200,
        rho=0.5,
        logitscale=1.0,
        y_dim=1,
        learn_rate=1e-2,
        epochs=20,
        batch_size=64,
        print_=False,
    )
    print("\n===== Backdoor Cocycle Smoke Test =====")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:>20}: {v:.6f}")
        else:
            print(f"{k:>20}: {v}")
