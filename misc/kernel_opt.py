import numpy as np
import torch
import torch.nn.functional as F

def kde_log_likelihood(X_train, X_val, lengthscale):
    """
    Compute the average log likelihood on X_val when estimating the density
    from X_train using a Gaussian kernel with the specified lengthscale.
    
    The density at a point x is estimated as:
      p(x) = 1/n * sum_{i=1}^n N(x; X_train[i], lengthscale^2)
    
    where N(x; μ, σ²) is the Gaussian density.
    """
    n_train = X_train.shape[0]
    # Compute pairwise differences: shape (n_val, n_train)
    differences = X_val.unsqueeze(1) - X_train.unsqueeze(0)
    # Compute the exponent term for the Gaussian density
    exponent = - (differences ** 2) / (2 * (lengthscale ** 2))
    
    # Constant term: -log(n_train) - log(sqrt(2π)*lengthscale)
    const_term = - torch.log(torch.tensor(n_train, dtype=torch.float32)) \
                 - 0.5 * torch.log(torch.tensor(2 * np.pi, dtype=torch.float32)) \
                 - torch.log(torch.tensor(lengthscale, dtype=torch.float32))
    
    # Compute log density using logsumexp for numerical stability
    log_density = const_term + torch.logsumexp(exponent, dim=1)
    return log_density.mean()  # Return average log likelihood over validation points

def k_fold_expected_log_likelihood(X, candidate_lengthscales, k=5):
    """
    Perform k-fold cross-validation on 1D data X to select the Gaussian kernel 
    lengthscale that maximizes the expected log likelihood.
    
    Args:
      X: A 1D torch.Tensor of data points.
      candidate_lengthscales: Iterable of candidate lengthscale values.
      k: Number of folds.
      
    Returns:
      best_lengthscale: The candidate with the highest average log likelihood.
      results: A dictionary mapping each candidate lengthscale to its CV log likelihood.
    """
    N = X.shape[0]
    indices = torch.randperm(N)
    fold_size = N // k
    results = {}
    
    for lengthscale in candidate_lengthscales:
        fold_ll = []
        for i in range(k):
            start = i * fold_size
            end = (i+1) * fold_size if i < k-1 else N
            val_indices = indices[start:end]
            train_indices = torch.cat([indices[:start], indices[end:]])
            X_val = X[val_indices]
            X_train = X[train_indices]
            ll = kde_log_likelihood(X_train, X_val, lengthscale)
            fold_ll.append(ll.item())
        avg_ll = np.mean(fold_ll)
        results[lengthscale] = avg_ll
        print(f"Lengthscale {lengthscale}: CV average log likelihood = {avg_ll:.4f}")
    
    best_lengthscale = max(results, key=results.get)
    return best_lengthscale, results

if __name__ == '__main__':
    # Generate synthetic 1D data for demonstration (e.g., 100 data points from a standard normal)
    torch.manual_seed(42)
    X_data = torch.randn(100, dtype=torch.float32)
    
    # Define candidate lengthscales to test
    candidate_ls = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    best_ls, cv_results = k_fold_expected_log_likelihood(X_data, candidate_ls, k=5)
    print("Optimal lengthscale:", best_ls)
