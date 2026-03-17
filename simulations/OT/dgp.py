import numpy as np
import torch

try:
    from .helpers import multivariate_laplace
except ImportError:
    from helpers import multivariate_laplace


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _means():
    return (
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
    )


def _covariances(corr=0.5, additive=True):
    if additive:
        identity = np.eye(2)
        return identity, identity, identity

    return (
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, -corr], [-corr, 1.0]]),
        np.array([[(1 + corr), 0.0], [0.0, (1 / (1 + corr))]]),
    )


def _sample_noise(size, seed, corr=0.5, multivariate_noise=False, dist="laplace"):
    rng = np.random.default_rng(seed)

    if multivariate_noise:
        if dist == "laplace":
            xi = multivariate_laplace(size=size, rng=seed, corr=corr)
        else:
            cov = np.ones((2, 2)) * corr + (1 - corr) * np.eye(2)
            xi = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=size)

        xi[:, 1] = xi[:, 0] + xi[:, 1]
        return xi

    if dist == "laplace":
        return rng.laplace(size=(size, 2))

    return rng.normal(size=(size, 2))


def generate_ot_data(
    seed,
    n=None,
    m=None,
    corr=0.5,
    additive=True,
    multivariate_noise=False,
    dist="laplace",
):
    n0 = n1 = n2 = n or 250
    if m is not None:
        n1 = m

    m0, m1, m2 = _means()
    S0, S1, S2 = _covariances(corr=corr, additive=additive)
    L0 = np.linalg.cholesky(S0)
    L1 = np.linalg.cholesky(S1)
    L2 = np.linalg.cholesky(S2)

    xi0 = _sample_noise(
        size=n0,
        seed=seed,
        corr=corr,
        multivariate_noise=multivariate_noise,
        dist=dist,
    )
    xi1 = _sample_noise(
        size=n1,
        seed=seed + 1,
        corr=corr,
        multivariate_noise=multivariate_noise,
        dist=dist,
    )
    xi2 = _sample_noise(
        size=n2,
        seed=seed + 2,
        corr=corr,
        multivariate_noise=multivariate_noise,
        dist=dist,
    )

    P0 = m0 + xi0 @ L0.T
    P1 = m1 + xi1 @ L1.T
    P2 = m2 + xi2 @ L2.T

    xi_cf = (P0 - m0) @ np.linalg.inv(L0).T
    Y1_cf = m1 + xi_cf @ L1.T
    Y2_cf = m2 + xi_cf @ L2.T

    X_obs = np.concatenate(
        [
            np.zeros(n0, dtype=np.int64),
            np.ones(n1, dtype=np.int64),
            2 * np.ones(n2, dtype=np.int64),
        ]
    )[:, None]
    Y_obs = np.concatenate([P0, P1, P2], axis=0)

    return {
        "n0": n0,
        "n1": n1,
        "n2": n2,
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "S0": S0,
        "S1": S1,
        "S2": S2,
        "L0": L0,
        "L1": L1,
        "L2": L2,
        "P0": P0,
        "P1": P1,
        "P2": P2,
        "Y1_cf": Y1_cf,
        "Y2_cf": Y2_cf,
        "X_obs": torch.tensor(X_obs, dtype=torch.float32),
        "Y_obs": torch.tensor(Y_obs, dtype=torch.float32),
    }


def generate_backdoor_binary_data(
    seed,
    n=500,
    rho=0.5,
    logitscale=1.0,
    y_dim=1,
):
    rng = np.random.default_rng(seed)

    cov_zy = np.array([[1.0, rho], [rho, 1.0]])
    xi_zy = rng.multivariate_normal(mean=np.zeros(2), cov=cov_zy, size=n)
    xi_x = rng.normal(size=n)

    z = xi_zy[:, 0]
    xi_y = xi_zy[:, 1]
    propensity = _sigmoid(logitscale * z)
    x = rng.binomial(1, propensity, size=n)

    if y_dim == 1:
        y0 = xi_y[:, None]
        y1 = (1.0 + xi_y)[:, None]
    elif y_dim == 2:
        y0 = np.column_stack((xi_y, xi_y))
        y1 = np.column_stack((1.0 + xi_y, 1.0 + xi_y))
    else:
        raise ValueError(f"Unsupported y_dim={y_dim}. Only 1 and 2 are currently supported.")

    y = y0.copy()
    y[x == 1] = y1[x == 1]

    x_column = x[:, None].astype(np.float32)
    z_column = z[:, None].astype(np.float32)
    inputs_x = x_column
    inputs_xz = np.concatenate([x_column, z_column], axis=1)
    outcomes = y.astype(np.float32)

    mask0 = x == 0
    mask1 = x == 1

    return {
        "n": n,
        "rho": rho,
        "logitscale": logitscale,
        "y_dim": y_dim,
        "xi_x": xi_x,
        "xi_y": xi_y,
        "Z": z,
        "X": x,
        "Y": y.squeeze(-1) if y_dim == 1 else y,
        "Y0": y0,
        "Y1": y1,
        "propensity": propensity,
        "X_obs": torch.tensor(inputs_x, dtype=torch.float32),
        "XZ_obs": torch.tensor(inputs_xz, dtype=torch.float32),
        "Y_obs": torch.tensor(outcomes, dtype=torch.float32),
        "Z_obs": torch.tensor(z_column, dtype=torch.float32),
        "X0": x[mask0],
        "X1": x[mask1],
        "Z0": z[mask0],
        "Z1": z[mask1],
        "Y_x0": y[mask0],
        "Y_x1": y[mask1],
    }
