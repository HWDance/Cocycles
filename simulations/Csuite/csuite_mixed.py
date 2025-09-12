import torch
from torch.distributions import Distribution, MixtureSameFamily, Categorical, Normal, Cauchy, Gamma, Bernoulli, Laplace
import torch.nn.functional as F

# Mixture dist with variance 1
means = torch.tensor([-3**0.5/2, 3**0.5/2])
scales = torch.full((2,), 0.5)
mixture = MixtureSameFamily(Categorical(torch.ones(2)/2), Normal(means, scales))

# Mixtures of different families
class Mixture1D:
    
    def __init__(self,base_dists,probabilities,noints,scales):
        self.dists = base_dists
        self.probabilities = probabilities
        self.noints = noints
        self.scales = scales
        
    def sample(self,size):
        C = OneHotCategorical(self.probabilities).sample(size)[:,0]
        Z = torch.zeros((size[0],len(self.probabilities)))
        for i in range(len(self.dists)):
            Z[:,i] = self.noints[i]+self.scales[i]*self.dists[i].sample(size).T
        return (Z*C).sum(1)[:,None]    


def _get_noises(
    N: int,
    d: int,
    seed: int | None = None
) -> list[torch.Tensor]:
    """
    Returns:
    --------
    A list of d tensors: [U₁, U₂, …, U_d], each of shape (N,1).
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Gen U1
    u1 = Normal(0,1).sample((N,1))

    # Gen rest
    dists = [Gamma(1,1), Bernoulli(0.5), Gamma(1,1), mixture]
    transforms = [lambda x : 1/x, lambda x : 2*x-1, lambda x : x, lambda x : x ]
    noises_rest = []
    for j in range(1,d):
        noises_rest.append(transforms[j-1](dists[j-1].sample((N,1))))

    return [u1] + noises_rest[::-1]


def _get_bd_noises(
    N: int,
    d: int,
    seed: int | None = None
) -> list[torch.Tensor]:
    """
    Returns:
    --------
    A list of d tensors: [U₁, U₂, …, U_d], each of shape (N,1).
    """
    if seed is not None:
        torch.manual_seed(seed)

    u_z = Normal(0,1).sample((N,d-2))
    u_x = torch.ones((N,1))

    means = torch.tensor([[-2, 0.0]]).T  # shape (2, 1)
    scales = torch.tensor([[-1.0, 1.0]]).T  # shape (2, 1)
    probabilities = torch.tensor([0.5, 0.5])  # mixture probabilities
    base_dists = [HalfNormal(1), HalfCauchy(1)]
    noise_dist = Mixture1D(base_dists, probabilities, means, scales)
    u_y = noise_dist.sample((N,1))


    return u_z,u_x,u_y


# -------------------- TWO-VARIABLE --------------------

def generate_2var_linear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    return_u: bool = False,
):
    """
    2-VAR linear SCM:
      u1; x1 = u1
      u2; x2 = x1 + u1
    noise_dists: list of 2 Distribution objects for u1,u2.
    noise_transform: applied to each sampled noise.
    """
    u1, u2 = _get_noises(N, 2, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)
    x2 = x1 + u2
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)
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
):
    """
    2-VAR nonlinear SCM:
      u1; x1 = u1
      u2; x2 = sin(x1) + u1
    """

    u1, u2 = _get_noises(N, 2, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)
    x2 = torch.sin(x1) + u2
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)
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
):
    """
    TRIANGLE linear SCM per eqns (76)-(78).
    """

    u1, u2, u3 = _get_noises(N, 3, seed)
    x1 = u1 + 1.0
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)
    x2 = 10.0 * x1 - u2
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)
    x3 = 0.5 * x2 + x1 + u3
    if intervention_node == 3:
        x3 = torch.full_like(x3, intervention_value)
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
):
    """
    TRIANGLE nonlinear SCM per eqns (79)-(81).
    """

    u1, u2, u3 = _get_noises(N, 3, seed)
    x1 = u1 + 1.0
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)
    x2 = 2.0 * x1**2 + u2
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)
    x3 = 20.0 * (1.0 + torch.exp(-x2**2 + x1)) + u3
    if intervention_node == 3:
        x3 = torch.full_like(x3, intervention_value)
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
):
    """
    FORK linear SCM per Fig.9(e).
    """

    u1, u2, u3, u4 = _get_noises(N, 4, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)
    x2 = 2.0 - u2
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)
    x3 = 0.25 * x2 - 1.5 * x1 + 0.5 * u3
    if intervention_node == 3:
        x3 = torch.full_like(x3, intervention_value)
    x4 = x3 + 0.25 * u4
    if intervention_node == 4:
        x4 = torch.full_like(x4, intervention_value)
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
):
    """
    FORK nonlinear SCM per Fig.9(f).
    """

    u1, u2, u3, u4 = _get_noises(N, 4, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)
    x2 = u2.clone()
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)
    x3 = 4.0 / (1.0 + torch.exp(-x1 - x2)) - x2**2 + 0.5 * u3
    if intervention_node == 3:
        x3 = torch.full_like(x3, intervention_value)
    x4 = 20.0 / (1.0 + torch.exp(0.5 * x3**2 - x3)) + u4
    if intervention_node == 4:
        x4 = torch.full_like(x4, intervention_value)
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
):
    """
    5-CHAIN linear SCM per Fig.9(c).
    """

    u1, u2, u3, u4, u5 = _get_noises(N, 5, seed)
    x1 = u1.clone()
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)
    x2 = 10.0 * x1 - u2
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)
    x3 = 0.25 * x2 + 2.0 * u3
    if intervention_node == 3:
        x3 = torch.full_like(x3, intervention_value)
    x4 = x3 + u4
    if intervention_node == 4:
        x4 = torch.full_like(x4, intervention_value)
    x5 = -x4 + u5
    if intervention_node == 5:
        x5 = torch.full_like(x5, intervention_value)
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
):
    """
    5-CHAIN nonlinear SCM (custom length-5).
    """

    u1, u2, u3, u4, u5 = _get_noises(N, 5, seed)
    x1 = torch.tanh(u1)
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)
    x2 = x1**2 + u2
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)
    x3 = torch.sin(x2) + u3
    if intervention_node == 3:
        x3 = torch.full_like(x3, intervention_value)
    x4 = x3 * x2 + u4
    if intervention_node == 4:
        x4 = torch.full_like(x4, intervention_value)
    x5 = torch.exp(-x4) + u5
    if intervention_node == 5:
        x5 = torch.full_like(x5, intervention_value)
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

def layer(parent: torch.Tensor, x_noise: torch.Tensor) -> torch.Tensor:
    """
    Implements soft truncation for both input and noise variables,
    approximately preserves mean=0 and var=1.
    """
    return F.softplus(parent + 1) + F.softplus(0.5 + x_noise) - 3.0

def generate_backdoor_linear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    loc: float = 0.0,
    scale: float = 1.0,
    return_u: bool = False,
):
    """
    Large backdoor SCM with all structural maps linear and binary treatment x7.

    Args:
        N:                  number of samples
        seed:               random seed (if not None)
        intervention_node:  which node (0…8) to intervene on (0-based indexing)
        intervention_value: value to force at the intervened node
        loc, scale:         noise parameters for all Normal base noises
        return_u:           if True, also return the matrix of raw noise terms

    Returns:
        X: tensor of shape (N, 9) with columns [x0, x1, …, x8]
        (optionally) U: tensor of shape (N, 9) with the raw noise inputs for each x
    """
    if seed is not None:
        torch.manual_seed(seed)

    # 0) x0_noise and x0 = 1.8 * u0 - 1
    u0 = torch.randn(N, 1) * scale + loc
    x0 = 1.8 * u0 - 1.0
    if intervention_node == 0:
        x0 = torch.full_like(x0, intervention_value)

    # 1) x1 ~ Normal(1.5 * x0, (0.25*scale)^2)
    mu1 = 1.5 * x0
    u1 = torch.randn_like(mu1) * (0.25 * scale)
    x1 = mu1 + u1
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)

    # 2) x2 = x0 + u2
    u2 = torch.randn(N, 1) * scale + loc
    x2 = x0 + u2
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)

    # 3) x3 = x1 + u3
    u3 = torch.randn(N, 1) * scale + loc
    x3 = x1 + u3
    if intervention_node == 3:
        x3 = torch.full_like(x3, intervention_value)

    # 4) x4 = x2 + u4
    u4 = torch.randn(N, 1) * scale + loc
    x4 = x2 + u4
    if intervention_node == 4:
        x4 = torch.full_like(x4, intervention_value)

    # 5) x5 = x3 + u5
    u5 = torch.randn(N, 1) * scale + loc
    x5 = x3 + u5
    if intervention_node == 5:
        x5 = torch.full_like(x5, intervention_value)

    # 6) x6 = x4 + u6
    u6 = torch.randn(N, 1) * scale + loc
    x6 = x4 + u6
    if intervention_node == 6:
        x6 = torch.full_like(x6, intervention_value)

    # 7) binary treatment x7: threshold of a linear precursor
    #    precursor mean = (x5 + 1)/1.5 - 1
    mu7 = (x5 + 1.0) / 1.5 - 1.0
    u7_cont = torch.randn(N, 1) * (0.15 * scale) + mu7
    x7 = (u7_cont > 0.5).float()
    if intervention_node == 7:
        x7 = torch.full_like(x7, intervention_value)

    # 8) final node x8 ~ Laplace(loc = (1.3*x6 - x7)/3 + 1, scale=0.6)
    loc8 = (1.3 * x6 - x7) / 3.0 + 1.0
    n_mix = 10
    mix = Categorical(1/torch.linspace(1,n_mix,n_mix))
    comp = Laplace(2**torch.linspace(1,n_mix,n_mix), 2**torch.linspace(1,n_mix,n_mix)/2**0.5)
    u8 = MixtureSameFamily(mix, comp).sample((N, 1))
    x8 = loc8 + u8
    if intervention_node == 8:
        x8 = torch.full_like(x8, intervention_value)

    # assemble outputs
    X = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8], dim=1)
    if return_u:
        U = torch.cat([u0, u1, u2, u3, u4, u5, u6, u7_cont, u8], dim=1)
        return X, U

    return X


def generate_backdoor_nonlinear(
    N: int,
    seed: int | None = None,
    intervention_node: int | None = None,
    intervention_value: float | None = None,
    loc: float = 0.0,
    scale: float = 1.0,
    return_u: bool = False,
):

    if seed is not None:
        torch.manual_seed(seed)

    # 0) x0_noise and x0
    u0 = torch.randn(N, 1) * scale + loc
    x0 = F.softplus(1.8 * u0) - 1
    if intervention_node == 0:
        x0 = torch.full_like(x0, intervention_value)

    # 1) x1 ~ Normal(1.5*layer(x0, 0), (0.25*scale)^2)
    mu1 = layer(x0, torch.zeros_like(x0)) * 1.5
    u1 = torch.randn_like(mu1) * (0.25 * scale)
    x1 = mu1 + u1
    if intervention_node == 1:
        x1 = torch.full_like(x1, intervention_value)

    # 2) x2 = layer(x0, x2_noise)
    u2 = torch.randn(N, 1) * scale + loc
    x2 = layer(x0, u2)
    if intervention_node == 2:
        x2 = torch.full_like(x2, intervention_value)

    # 3) x3 = layer(x1, x3_noise)
    u3 = torch.randn(N, 1) * scale + loc
    x3 = layer(x1, u3)
    if intervention_node == 3:
        x3 = torch.full_like(x3, intervention_value)

    # 4) x4 = layer(x2, x4_noise)
    u4 = torch.randn(N, 1) * scale + loc
    x4 = layer(x2, u4)
    if intervention_node == 4:
        x4 = torch.full_like(x4, intervention_value)

    # 5) x5 = layer(x3, x5_noise)
    u5 = torch.randn(N, 1) * scale + loc
    x5 = layer(x3, u5)
    if intervention_node == 5:
        x5 = torch.full_like(x5, intervention_value)

    # 6) x6 = layer(x4, x6_noise)
    u6 = torch.randn(N, 1) * scale + loc
    x6 = layer(x4, u6)
    if intervention_node == 6:
        x6 = torch.full_like(x6, intervention_value)

    # 7) binary treatment x7 via thresholded LogNormal
    mean7 = F.softplus(x5 + 1) / 1.5 - 1
    z7 = torch.randn(N, 1) * (0.15 * scale) + mean7
    u7_cont = torch.exp(z7)               # LogNormal sample
    x7 = (u7_cont > 0.5).float()          # threshold to {0,1}
    if intervention_node == 7:
        x7 = torch.full_like(x7, intervention_value)

    # 8) final node x8 ~ Laplace(loc=-softplus((-1.3 x6 + x7)/3 + 1)+2, scale=0.6)
    loc8 = -F.softplus((-1.3 * x6 + x7) / 3 + 1) + 2

    n_mix = 10
    mix = Categorical(1/torch.linspace(1,n_mix,n_mix))
    comp = Laplace(2**torch.linspace(1,n_mix,n_mix), 2**torch.linspace(1,n_mix,n_mix)/2**0.5)
    u8 = MixtureSameFamily(mix, comp).sample((N, 1))
    x8 = loc8 + u8
    if intervention_node == 8:
        x8 = torch.full_like(x8, intervention_value)

    # stack into data matrix
    X = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7, x8], dim=1)

    if return_u:
        # record raw noises (u7_cont for x7, u8 for x8)
        U = torch.cat([u0, u1, u2, u3, u4, u5, u6, u7_cont, u8], dim=1)
        return X, U

    return X
