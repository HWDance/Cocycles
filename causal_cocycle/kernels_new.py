import torch
import torch.nn as nn

class Kernel(nn.Module):
    def __init__(self, lengthscale=None, scale=None):
        super().__init__()
        # Default: learnable scalar lengthscale
        if lengthscale is None:
            self.log_lengthscale = nn.Parameter(torch.tensor([0.0]))  # log(1.0)
        else:
            self.log_lengthscale = nn.Parameter(torch.log(lengthscale))

        if scale is None:
            self.scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        else:
            self.scale = nn.Parameter(scale, requires_grad=False)

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)

    def get_gram(self, X, Z):
        raise NotImplementedError


class GaussianKernel(Kernel):
    def get_gram(self, X, Z):
        X_scaled = X / self.lengthscale
        Z_scaled = Z / self.lengthscale
        distsq = torch.cdist(X_scaled, Z_scaled, p=2.0) ** 2
        return self.scale * torch.exp(-0.5 * distsq)


class ExponentialKernel(Kernel):
    def get_gram(self, X, Z):
        X_scaled = X / self.lengthscale
        Z_scaled = Z / self.lengthscale
        dist = torch.cdist(X_scaled, Z_scaled, p=2.0)
        return self.scale * torch.exp(-0.5 * dist)


class InverseGaussianKernel(Kernel):
    def get_gram(self, X, Z):
        X_scaled = X * self.lengthscale
        Z_scaled = Z * self.lengthscale
        distsq = torch.cdist(X_scaled, Z_scaled, p=2.0) ** 2
        return self.scale * torch.exp(-0.5 * distsq)


class MultivariateGaussianKernel(Kernel):
    """
    lengthscale is expected to be a d x d matrix (e.g., a whitening transform)
    """
    def get_gram(self, X, Z):
        X_trans = X @ self.lengthscale
        Z_trans = Z @ self.lengthscale
        distsq = torch.cdist(X_trans, Z_trans, p=2.0) ** 2
        return self.scale * torch.exp(-0.5 * distsq)


class LinearKernel(Kernel):
    def get_gram(self, X, Z):
        return X @ Z.T


def median_heuristic(X):
    """
    Median heuristic for Gaussian bandwidth (sqrt of median pairwise squared distance / 2)
    """
    with torch.no_grad():
        D = torch.cdist(X, X, p=2.0) ** 2
        mask = torch.tril(torch.ones_like(D, dtype=torch.bool), -1)
        med = D[mask].median()
        return (med / 2).sqrt()

def median_heuristic_ard(X):
    """
    ARD lengthscales normalized to match isotropic heuristic scale.
    
    Returns:
    - lengthscales: (d,) tensor
    """
    n, d = X.shape
    ard_ls = torch.zeros(d)
    for j in range(d):
        diffs = X[:, j].unsqueeze(0) - X[:, j].unsqueeze(1)
        distsq = (diffs ** 2)[torch.triu_indices(n, n, offset=1).unbind()]
        ard_ls[j] = (distsq.median() / 2).sqrt()

    # Compute isotropic heuristic
    pairwise_sq = torch.cdist(X, X, p=2.0).pow(2)
    isotropic_med = (pairwise_sq[torch.triu_indices(n, n, offset=1).unbind()].median() / 2).sqrt()

    norm = ard_ls.norm(p=2)
    lengthscales = ard_ls * (isotropic_med / norm)
    return lengthscales

