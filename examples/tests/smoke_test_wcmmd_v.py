import torch

from causal_cocycle.kernels import gaussian_kernel
from causal_cocycle.kernels_new import GaussianKernel
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.rkhs_functionals import KRRFunctional
from causal_cocycle.rkhs_weight_estimation import RKHSWeightEstimator


def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2))


class DummyModel(torch.nn.Module):
    def cocycle_outer(self, inputs1, inputs2, outputs):
        x = inputs1["X"]
        z = inputs1["Z"]
        n = outputs.shape[0]
        base = outputs.unsqueeze(0).expand(n, n, outputs.shape[-1])
        shift = 0.01 * x.unsqueeze(1)[..., : outputs.shape[-1]] + 0.01 * z[:, :1].unsqueeze(1)
        return base + shift


def main():
    torch.manual_seed(0)

    n_train = 40
    n_test = 20
    x_dim = 1
    z_dim = 2
    y_dim = 1

    X_train = torch.randn(n_train, x_dim)
    Z_train = torch.randn(n_train, z_dim)
    noise_train = 0.05 * torch.randn(n_train, y_dim)
    Y_train = 0.8 * X_train + 0.3 * Z_train[:, :1] - 0.2 * Z_train[:, 1:2] + noise_train

    X_test = torch.randn(n_test, x_dim)
    Z_test = torch.randn(n_test, z_dim)
    noise_test = 0.05 * torch.randn(n_test, y_dim)
    Y_test = 0.8 * X_test + 0.3 * Z_test[:, :1] - 0.2 * Z_test[:, 1:2] + noise_test

    U_test = torch.cat([X_test, Z_test], dim=-1)

    kernel_u = GaussianKernel(lengthscale=torch.ones(x_dim + z_dim))
    kernel_y = GaussianKernel(lengthscale=torch.ones(y_dim))
    functional = KRRFunctional(kernel_u, penalty=1e-1)
    estimator = RKHSWeightEstimator(functional, kernel_y=kernel_y)

    estimator.fit(X_train, Z_train, Y_train)
    Y_pred_before = estimator.functional.predict(U_test)
    rmse_before = rmse(Y_pred_before, Y_test)

    cv_losses = estimator.tune(
        X_train,
        Z_train,
        Y_train,
        maxiter=5,
        nfold=3,
        learn_rate=5e-2,
        print_=False,
    )
    estimator.fit(X_train, Z_train, Y_train)
    Y_pred_after = estimator.functional.predict(U_test)
    rmse_after = rmse(Y_pred_after, Y_test)

    W = estimator.training_weights()
    assert W.shape == (n_train, n_train), W.shape
    assert torch.isfinite(W).all()

    model = DummyModel()
    loss_factory = CocycleLossFactory([gaussian_kernel(), gaussian_kernel()])
    loss = loss_factory.build_loss(
        "WCMMD_V",
        X={"X": X_train, "Z": Z_train},
        Y=Y_train,
        weight_estimator=estimator,
        weight_mode="fixed",
    )
    loss_value = loss(model, {"X": X_train, "Z": Z_train}, Y_train)
    assert torch.isfinite(loss_value), loss_value

    print("smoke_test_ok")
    print("rmse_before_cv", float(rmse_before))
    print("rmse_after_cv", float(rmse_after))
    print("cv_losses", [float(x) for x in cv_losses])
    print("weights_shape", tuple(W.shape))
    print("loss_value", float(loss_value))


if __name__ == "__main__":
    main()
