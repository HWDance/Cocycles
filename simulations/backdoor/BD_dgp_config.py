import torch
from torch.distributions import Normal, HalfNormal, HalfCauchy, Bernoulli
from BD import Mixture1D

"""DGP configs"""
# Create coefficients: 1/linspace(1, D, D)**1, normalized to sum to 1.

D = 10
coeffs = 1 / torch.linspace(1, D, D)[:, None] ** 1
coeffs *= 1 / coeffs.sum()

# Mixture parameters for the base distribution of U.
means = torch.tensor([[-2, 0.0]]).T  # shape (2, 1)
scales = torch.tensor([[-1.0, 1.0]]).T  # shape (2, 1)
probabilities = torch.tensor([0.5, 0.5])  # mixture probabilities
base_dists = [HalfNormal(1), HalfCauchy(1)]
noise_dist = Mixture1D(base_dists, probabilities, means, scales)
noise_dist = Mixture1D(base_dists, probabilities, means, scales)


# Global model configuration dictionary.
dgp_config = {
    "N": 10000,            
    "D": D,             
    "projection_coeffs": coeffs,
    "covariate_corr" : 0.0,
    "covariate_dist" : Normal(0,1.5),
    "noise_dist" : noise_dist,
}
