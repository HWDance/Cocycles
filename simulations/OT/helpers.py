import numpy as np
import torch

def multivariate_laplace(mu = np.zeros(2), corr =0.25, b=1.0, size=1, rng=None):
    """
    Sample from a multivariate Laplace distribution using the
    Gaussian-exponential mixture representation.

    Parameters
    ----------
    mu : array_like, shape (d,)
        Mean vector.
    Sigma : array_like, shape (d, d)
        Covariance matrix (positive definite).
    b : float
        Scale parameter (Laplace 'spread').
    size : int
        Number of samples.
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    samples : ndarray, shape (size, d)
        Multivariate Laplace samples.
    """
    rng = np.random.default_rng(rng)
    mu = np.asarray(mu)
    d = mu.shape[0]

    # Cholesky of covariance
    Sigma = np.ones((d,d))*corr + (1-corr)*np.eye(d)
    L = np.linalg.cholesky(Sigma)

    # Exponential mixture variable
    W = rng.exponential(scale=b, size=size)

    # Multivariate normal samples
    Z = rng.normal(size=(size, d)) @ L.T

    # Combine
    return mu + np.sqrt(W)[:, None] * Z

def compute_weights_from_kernel(kernel, X1, x1_query):
    """
    Returns normalized weights w_i(x1_query) using kernel.get_gram().
    X1 is a 1D tensor of training values.s
    x1_query is a scalar (or 0D tensor).
    """
    if x1_query.dim() == 0:
        x1_query = x1_query.view(1, 1)
    elif x1_query.dim() == 1:
        x1_query = x1_query.unsqueeze(0)
    if X1.dim() == 1:
        X1 = X1.unsqueeze(1)
    K_vals = kernel.get_gram(x1_query, X1).flatten()
    return K_vals / K_vals.sum()

# Modified KR transport (1D) with provided weights
def build_KR_map(X: torch.Tensor, Y: torch.Tensor, 
                w_src: torch.Tensor = None, w_tgt: torch.Tensor = None, epsilon: float = 1e-8):

    # Get weights if not provided
    if w_src is None:
        n_src = X.numel()
        w_src = torch.ones(n_src, device=X.device) / n_src
    if w_tgt is None:
        n_tgt = Y.numel()
        w_tgt = torch.ones(n_tgt, device=Y.device) / n_tgt
    
    # Sort X and Y first
    X_sorted_full, idx_X = torch.sort(X)
    Y_sorted_full, idx_Y = torch.sort(Y)
    w_src_sorted_full = w_src[idx_X]
    w_tgt_sorted_full = w_tgt[idx_Y]
    
    # Get unique sorted values for X, and sum the weights of duplicates.
    X_sorted, inverse_idx_X = torch.unique(X_sorted_full, sorted=True, return_inverse=True)
    w_src_sorted = torch.zeros_like(X_sorted, dtype=w_src.dtype)
    w_src_sorted = w_src_sorted.scatter_add_(0, inverse_idx_X, w_src_sorted_full)
    
    # Do the same for Y.
    Y_sorted, inverse_idx_Y = torch.unique(Y_sorted_full, sorted=True, return_inverse=True)
    w_tgt_sorted = torch.zeros_like(Y_sorted, dtype=w_tgt.dtype)
    w_tgt_sorted = w_tgt_sorted.scatter_add_(0, inverse_idx_Y, w_tgt_sorted_full)
    
    # Normalize the weights.
    w_src_sorted = w_src_sorted / w_src_sorted.sum()
    w_tgt_sorted = w_tgt_sorted / w_tgt_sorted.sum()

    def S(z):
        z = z.double()
        if epsilon == 0:
            return (z>=0).float()
        else:
            return torch.where(z < -epsilon, torch.zeros_like(z),
                               torch.where(z > epsilon, torch.ones_like(z),
                                           (z + epsilon) / (2 * epsilon)))
    def F_src(y):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=X_sorted.device)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        X_exp = X_sorted.unsqueeze(0)
        S_vals = S(y - X_exp)
        return torch.sum(w_src_sorted * S_vals, dim=-1)
    def Q_tgt(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=Y_sorted.device)
        if t.dim() > 1:
            t = t.squeeze(-1)
        cumsum = torch.cumsum(w_tgt_sorted, dim=0)
        indices = torch.searchsorted(cumsum, t.unsqueeze(1)).squeeze(1)
        indices = torch.clamp(indices, 0, Y_sorted.numel()-1)
        j = indices-1
        j_next = indices
        cumsum = torch.concatenate((torch.zeros(1),cumsum))
        cumsum_j = cumsum[j+1]
        w_j_next = w_tgt_sorted[j_next]
        s = (t - cumsum_j) / w_j_next
        Y_j_next = Y_sorted[j_next].double()
        Y_prev = Y_sorted[torch.clamp(j, min=0)].double()
        return Y_j_next - epsilon + 2 * epsilon * s        
    def KR(y):
        t_val = F_src(y)
        return Q_tgt(t_val)
    return KR