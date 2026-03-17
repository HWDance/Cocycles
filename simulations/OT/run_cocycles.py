import torch
from causal_cocycle.model_new import ZukoCocycleModel
from causal_cocycle.optimise_new import validate
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.kernels import gaussian_kernel
from architectures import get_anchored_discrete_flows_single
from dgp import generate_ot_data

# === Step 1: Build ZukoCocycleModels ===

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

# === Step 2: Loss and kernel ===

def build_loss(X, Y):
    kernel = [gaussian_kernel(), gaussian_kernel()]
    loss_factory = CocycleLossFactory(kernel)
    return loss_factory.build_loss("CMMD_U", X, Y)

# === Step 3: Run validation + evaluation ===

def run(seed=0, n=250, learn_rate = 1e-3, wrong_order = False, corr = 0.5, additive = True, multivariate_noise = False, dist = "laplace", epochs=1000, print_=True):
    data = generate_ot_data(
        seed=seed,
        n=n,
        corr=corr,
        additive=additive,
        multivariate_noise=multivariate_noise,
        dist=dist,
    )
    X_raw = data["X_obs"]
    Y = data["Y_obs"]
    P0 = Y[:n]
    true_Y1_cf = torch.tensor(data["Y1_cf"], dtype=torch.float32)
    true_Y2_cf = torch.tensor(data["Y2_cf"], dtype=torch.float32)

    if wrong_order:
        Y = Y.flip(dims=[1])
        P0 = P0.flip(dims=[1])
        true_Y1_cf = true_Y1_cf.flip(dims=[1])
        true_Y2_cf = true_Y2_cf.flip(dims=[1])

    x_dim = 1
    y_dim = Y.shape[1]
    # Shuffle data
    perm = torch.randperm(X_raw.size(0))
    X_tr = X_raw[perm]
    Y_tr = Y[perm]

    opt_kwargs = dict(
        epochs=epochs,
        learn_rate=learn_rate,
        scheduler=False,
        batch_size=128,
        print_=print_,
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
        x0 = torch.zeros((n, 1))
        x1 = torch.ones((n, 1))
        x2 = 2 * torch.ones((n, 1))

        Y1_cf = best_model.cocycle(x1, x0, P0)
        Y2_cf = best_model.cocycle(x2, x0, P0)

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
    result = run(
        seed=0,
        n=50,
        learn_rate=1e-2,
        corr=0.5,
        additive=False,
        multivariate_noise=False,
        epochs=20,
        print_=False,
    )
    print("\n===== Cocycle Smoke Test =====")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k:>20}: {v:.6f}")
        else:
            print(f"{k:>20}: {v}")
