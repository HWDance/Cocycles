# Helper functions
import torch
from torch import nn
from torch.distributions import Normal,Uniform

class mmd:
    
    def __init__(self,kernel):
        self.kernel = kernel
    
    def get_median_heuristic(self,inputs,subsamples = 10000):
        """
        Returns median heuristic lengthscale for Gaussian kernel
        """
        
        inputs_batch = inputs[:subsamples]
        
        # Median heurstic for inputs
        Dist = torch.cdist(inputs_batch,inputs_batch, p = 2.0)**2
        Lower_tri = torch.tril(Dist, diagonal=-1).view(len(inputs_batch)**2).sort(descending = True)[0]
        Lower_tri = Lower_tri[Lower_tri!=0]
        self.kernel.log_lengthscale = nn.Parameter((Lower_tri.median()/2).sqrt().log())
            
        return
        
    def __call__(self,X,Y,median_heuristic = True,mmd_samples = 5000,heuristic_samples = 1000):
        if median_heuristic:
            self.get_median_heuristic(X,heuristic_samples)
        return (self.kernel.get_gram(X[:mmd_samples],X[:mmd_samples]).mean() 
                + self.kernel.get_gram(Y[:mmd_samples],Y[:mmd_samples]).mean()
                -2*self.kernel.get_gram(X[:mmd_samples],Y[:mmd_samples]).mean())**0.5

def kolmogorov_distance(X, Y):
    """
    Compute the two-sample Kolmogorov distance between samples X and Y.
    
    This is the maximum absolute difference between their empirical CDFs.
    """
    X_sorted = sorted(X)
    Y_sorted = sorted(Y)

    n_x, n_y = len(X_sorted), len(Y_sorted)
    i = j = 0      # Pointers
    cdf_x = cdf_y = 0.0
    max_diff = 0.0

    # "Walk" through both sorted samples
    while i < n_x and j < n_y:
        # Whichever sample has the smaller current value advances
        if X_sorted[i] < Y_sorted[j]:
            i += 1
            cdf_x = i / n_x
        elif X_sorted[i] > Y_sorted[j]:
            j += 1
            cdf_y = j / n_y
        else:
            # Values are equal, so increment both
            i += 1
            j += 1
            cdf_x = i / n_x
            cdf_y = j / n_y

        diff = abs(cdf_x - cdf_y)
        if diff > max_diff:
            max_diff = diff

    # If one sample is exhausted, the other might still have elements left
    while i < n_x:
        i += 1
        cdf_x = i / n_x
        diff = abs(cdf_x - cdf_y)
        if diff > max_diff:
            max_diff = diff

    while j < n_y:
        j += 1
        cdf_y = j / n_y
        diff = abs(cdf_x - cdf_y)
        if diff > max_diff:
            max_diff = diff

    return max_diff


class likelihood_loss:
    
    def __init__(self,dist, tail_adapt = False,tail_init = 10.0,log_det = True):
        self.dist = dist
        self.tail_adapt = tail_adapt
        self.parameters = torch.tensor([tail_init]).requires_grad_(True)
        self.log_det = log_det
    
    def __call__(self,model,inputs,outputs):
        if self.log_det:
            U,logdet = model.inverse_transformation(inputs,outputs)
            if self.tail_adapt:
                self.dist.df = self.parameters.abs()
            return torch.mean(-self.dist.log_prob(U) - logdet)
        else:
            U = model.inverse_transformation(inputs,outputs)
            if self.tail_adapt:
                self.dist.df = self.parameters.abs()
            return torch.mean(-self.dist.log_prob(U))
    
class propensity_score:
    
    def __init__(self,P,policy):
        self.P = P
        self.policy = policy
    """
    Pij = P(X = i |X* = j) is conditional dist on assignments `post error'
    given initial policy assignment
    """
    def __call__(self,X,V):
        assert(len(X) == len(V))
        col_select = self.policy(V,flip_prob = 0)[:,0]
        row_select = torch.linspace(0,len(X)-1,len(X)).int()
        conditional_dists = (self.P[...,None] @ torch.ones((1,1,len(V))))[:,col_select,row_select]
        return conditional_dists[X[:,0].int(),row_select]
    


def ks_statistic(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    a_s, _ = torch.sort(a); b_s, _ = torch.sort(b)
    all_vs = torch.cat([a_s, b_s]).unique()
    cdf_a = torch.bucketize(all_vs, a_s, right=True).float() / a_s.numel()
    cdf_b = torch.bucketize(all_vs, b_s, right=True).float() / b_s.numel()
    return (torch.abs(cdf_a - cdf_b).max()).item()


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(((a - b)**2).mean()).item()

def wasserstein1_repeat(xs, ys):
    """
    Compute the 1D Wasserstein-1 distance between two empirical distributions,
    where len(xs) = m, len(ys) = n, and n = K * m for some integer K.
    
    This repeats each x_i K times, sorts both lists, and returns the average L1 difference.
    
    Parameters
    ----------
    xs : array_like
        1D array of m samples from the first distribution.
    ys : array_like
        1D array of n = K * m samples from the second distribution.
    
    Returns
    -------
    w1 : float
        The Wasserstein-1 distance.
    """
    
    m = len(xs)
    n = len(ys)
    K = n // m

    if n % m != 0:
        raise ValueError("Sample sizes must satisfy n = K * m for some integer K.")

    # Repeat each x_i K times
    xs_rep = torch.repeat_interleave(xs, K)
    
    # Sort both
    xs_sorted = torch.sort(xs_rep)[0]
    ys_sorted = torch.sort(ys)[0]

    # Compute mean absolute difference
    w1 = torch.mean(torch.abs(xs_sorted - ys_sorted))
    return w1
