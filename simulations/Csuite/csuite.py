import torch
from torch.distributions import Distribution


def _get_noises(
    N: int,
    d: int,
    noise_dists: list[Distribution],
    noise_transforms: list[callable],
    seed: int | None = None
) -> list[torch.Tensor]:
    """
    Generalized noise‐generator for any SCM of dimension d ≥ 1, where:
      • noise_dists[0]: Distribution for U₁ (univariate)
      • noise_dists[1]: Distribution for “base” of (U₂…U_d)
          (univariate, from which we sample shape (N, d–1))
      • noise_transforms[0]: callable mapping a tensor (N,) → (N,1) for U₁
      • noise_transforms[1]: callable mapping a tensor (N, d–1) → (N, d–1)
          to induce correlation among (U₂…U_d)

    Arguments:
    ----------
    - N               : int, number of samples
    - d               : int ≥ 1, total number of variables in SCM
    - noise_dists     : list of length 2:
        [dist_u1, dist_base]
        • dist_u1  : univariate Distribution for U₁
        • dist_base: univariate Distribution, to be sampled with .sample((N, d–1))
    - noise_transforms: list of length 2:
        [tf_u1, tf_rest]
        • tf_u1   : callable mapping (N,) → (N,1)
        • tf_rest : callable mapping (N, d–1) → (N, d–1)
    - seed            : optional RNG seed

    Returns:
    --------
    A list of d tensors: [U₁, U₂, …, U_d], each of shape (N,1).
    """
    if seed is not None:
        torch.manual_seed(seed)

    if d < 1:
        raise ValueError("d must be at least 1")
    if len(noise_dists) != 2 or len(noise_transforms) != 2:
        raise ValueError("noise_dists and noise_transforms must be lists of length 2")

    dist_u1, dist_base = noise_dists
    tf_u1, tf_rest     = noise_transforms

    # 1) Draw U₁
    raw1 = dist_u1.sample((N,))            # → (N,)
    u1   = tf_u1(raw1).view(N, 1)          # → (N,1)

    if d == 1:
        return [u1]

    # 2) Draw “base” noise for (U₂…U_d) as shape (N, d–1)
    raw_rest = dist_base.sample((N, d - 1))  # → (N, d–1)

    # 3) Apply tf_rest to induce correlation (still (N, d–1))
    rest = tf_rest(raw_rest)
    if rest.dim() != 2 or rest.size(1) != (d - 1):
        raise ValueError(f"tf_rest must return a tensor of shape (N, {d-1})")

    # 4) Split rest into d–1 individual columns of shape (N,1)
    noises = [u1]
    for i in range(d - 1):
        noises.append(rest[:, i : i+1])   # each → (N,1)

    return noises


# -------------------- TWO-VARIABLE --------------------

def generate_2var_linear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
    noise_dists: list[Distribution] = None,
    noise_transforms: list[callable] = None,
    intervention_fn: callable = torch.full_like,
):
    if noise_dists is None or len(noise_dists) != 2:
        raise ValueError("noise_dists must be a list of length 2")
    u1, u2 = _get_noises(N, 2, noise_dists, noise_transforms, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = intervention_fn(x1, intervention_value)
    x2 = x1 + u2
    if intervention_node == 2:
        x2 = intervention_fn(x2, intervention_value)
    X = torch.cat([x1, x2], dim=1)
    if return_u:
        return X, torch.cat([u1, u2], dim=1)
    return X


def generate_2var_nonlinear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
    noise_dists: list[Distribution] = None,
    noise_transforms: list[callable] = None,
    intervention_fn: callable = torch.full_like,
):
    if noise_dists is None or len(noise_dists) != 2:
        raise ValueError("noise_dists must be a list of length 2")
    u1, u2 = _get_noises(N, 2, noise_dists, noise_transforms, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = intervention_fn(x1, intervention_value)
    x2 = torch.sin(x1) + u2
    if intervention_node == 2:
        x2 = intervention_fn(x2, intervention_value)
    X = torch.cat([x1, x2], dim=1)
    if return_u:
        return X, torch.cat([u1, u2], dim=1)
    return X

# -------------------- TRIANGLE --------------------

def generate_triangle_linear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
    noise_dists: list[Distribution] = None,
    noise_transforms: list[callable] = None,
    intervention_fn: callable = torch.full_like,
):
    if noise_dists is None or len(noise_dists) != 2:
        raise ValueError("noise_dists must be a list of length 2")
    u1, u2, u3 = _get_noises(N, 3, noise_dists, noise_transforms, seed)
    x1 = u1 + 1.0
    if intervention_node == 1:
        x1 = intervention_fn(x1, intervention_value)
    x2 = 10.0 * x1 - u2
    if intervention_node == 2:
        x2 = intervention_fn(x2, intervention_value)
    x3 = 0.5 * x2 + x1 + u3
    if intervention_node == 3:
        x3 = intervention_fn(x3, intervention_value)
    X = torch.cat([x1, x2, x3], dim=1)
    if return_u:
        return X, torch.cat([u1, u2, u3], dim=1)
    return X


def generate_triangle_nonlinear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
    noise_dists: list[Distribution] = None,
    noise_transforms: list[callable] = None,
    intervention_fn: callable = torch.full_like,
):
    if noise_dists is None or len(noise_dists) != 2:
        raise ValueError("noise_dists must be a list of length 2")
    u1, u2, u3 = _get_noises(N, 3, noise_dists, noise_transforms, seed)
    x1 = u1 + 1.0
    if intervention_node == 1:
        x1 = intervention_fn(x1, intervention_value)
    x2 = 2.0 * x1**2 + u2
    if intervention_node == 2:
        x2 = intervention_fn(x2, intervention_value)
    x3 = 20.0 * (1.0 + torch.exp(-x2**2 + x1)) + u3
    if intervention_node == 3:
        x3 = intervention_fn(x3, intervention_value)
    X = torch.cat([x1, x2, x3], dim=1)
    if return_u:
        return X, torch.cat([u1, u2, u3], dim=1)
    return X

# ---------------------- FORK ----------------------

def generate_fork_linear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
    noise_dists: list[Distribution] = None,
    noise_transforms: list[callable] = None,
    intervention_fn: callable = torch.full_like,
):
    if noise_dists is None or len(noise_dists) != 2:
        raise ValueError("noise_dists must be a list of length 2")
    u1, u2, u3, u4 = _get_noises(N, 4, noise_dists, noise_transforms, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = intervention_fn(x1, intervention_value)
    x2 = 2.0 - u2
    if intervention_node == 2:
        x2 = intervention_fn(x2, intervention_value)
    x3 = 0.25 * x2 - 1.5 * x1 + 0.5 * u3
    if intervention_node == 3:
        x3 = intervention_fn(x3, intervention_value)
    x4 = x3 + 0.25 * u4
    if intervention_node == 4:
        x4 = intervention_fn(x4, intervention_value)
    X = torch.cat([x1, x2, x3, x4], dim=1)
    if return_u:
        return X, torch.cat([u1, u2, u3, u4], dim=1)
    return X


def generate_fork_nonlinear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
    noise_dists: list[Distribution] = None,
    noise_transforms: list[callable] = None,
    intervention_fn: callable = torch.full_like,
):
    if noise_dists is None or len(noise_dists) != 2:
        raise ValueError("noise_dists must be a list of length 2")
    u1, u2, u3, u4 = _get_noises(N, 4, noise_dists, noise_transforms, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = intervention_fn(x1, intervention_value)
    x2 = u2.clone()
    if intervention_node == 2:
        x2 = intervention_fn(x2, intervention_value)
    x3 = 4.0 / (1.0 + torch.exp(-x1 - x2)) - x2**2 + 0.5 * u3
    if intervention_node == 3:
        x3 = intervention_fn(x3, intervention_value)
    x4 = 20.0 / (1.0 + torch.exp(0.5 * x3**2 - x3)) + u4
    if intervention_node == 4:
        x4 = intervention_fn(x4, intervention_value)
    X = torch.cat([x1, x2, x3, x4], dim=1)
    if return_u:
        return X, torch.cat([u1, u2, u3, u4], dim=1)
    return X

# -------------------- 5-CHAIN --------------------

def generate_chain5_linear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
    noise_dists: list[Distribution] = None,
    noise_transforms: list[callable] = None,
    intervention_fn: callable = torch.full_like,
):
    if noise_dists is None or len(noise_dists) != 2:
        raise ValueError("noise_dists must be a list of length 2")
    u1, u2, u3, u4, u5 = _get_noises(N, 5, noise_dists, noise_transforms, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = intervention_fn(x1, intervention_value)
    x2 = 10.0 * x1 - u2
    if intervention_node == 2:
        x2 = intervention_fn(x2, intervention_value)
    x3 = 0.25 * x2 + 2.0 * u3
    if intervention_node == 3:
        x3 = intervention_fn(x3, intervention_value)
    x4 = x3 + u4
    if intervention_node == 4:
        x4 = intervention_fn(x4, intervention_value)
    x5 = -x4 + u5
    if intervention_node == 5:
        x5 = intervention_fn(x5, intervention_value)
    X = torch.cat([x1, x2, x3, x4, x5], dim=1)
    if return_u:
        return X, torch.cat([u1, u2, u3, u4, u5], dim=1)
    return X


def generate_chain5_nonlinear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
    noise_dists: list[Distribution] = None,
    noise_transforms: list[callable] = None,
    intervention_fn: callable = torch.full_like,
):
    if noise_dists is None or len(noise_dists) != 2:
        raise ValueError("noise_dists must be a list of length 2")
    u1, u2, u3, u4, u5 = _get_noises(N, 5, noise_dists, noise_transforms, seed)
    x1 = torch.tanh(u1)
    if intervention_node == 1:
        x1 = intervention_fn(x1, intervention_value)
    x2 = x1**2 + u2
    if intervention_node == 2:
        x2 = intervention_fn(x2, intervention_value)
    x3 = torch.sin(x2) + u3
    if intervention_node == 3:
        x3 = intervention_fn(x3, intervention_value)
    x4 = x3 * x2 + u4
    if intervention_node == 4:
        x4 = intervention_fn(x4, intervention_value)
    x5 = torch.exp(-x4) + u5
    if intervention_node == 5:
        x5 = intervention_fn(x5, intervention_value)
    X = torch.cat([x1, x2, x3, x4, x5], dim=1)
    if return_u:
        return X, torch.cat([u1, u2, u3, u4, u5], dim=1)
    return X


SCMS = {
    '2var_linear':         generate_2var_linear,
    '2var_nonlinear':      generate_2var_nonlinear,
    'triangle_linear':     generate_triangle_linear,
    'triangle_nonlinear':  generate_triangle_nonlinear,
    'fork_linear':         generate_fork_linear,
    'fork_nonlinear':      generate_fork_nonlinear,
    'chain5_linear':       generate_chain5_linear,
    'chain5_nonlinear':    generate_chain5_nonlinear,
}

SCM_DIMS = {
    '2var_linear': 2, '2var_nonlinear': 2,
    'triangle_linear': 3, 'triangle_nonlinear': 3,
    'fork_linear': 4, 'fork_nonlinear': 4,
    'chain5_linear': 5, 'chain5_nonlinear': 5,
}

SCM_MASKS = {
    '2var_linear': torch.tensor([
        [0, 0],
        [1, 0],
    ]),
    '2var_nonlinear': torch.tensor([
        [0, 0],
        [1, 0],
    ]),
    'triangle_linear': torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ]),
    'triangle_nonlinear': torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ]),
    'fork_linear': torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
    ]),
    'fork_nonlinear': torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
    ]),
    'chain5_linear': torch.tensor([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    'chain5_nonlinear': torch.tensor([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
}

