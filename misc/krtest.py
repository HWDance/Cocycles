import torch

def build_modified_KR_transport(Y: torch.Tensor, w_src: torch.Tensor, w_tgt: torch.Tensor, epsilon: float):
    """
    Builds the modified KR transport from P_{Y|x1} to P_{Y|x2} using ε-smoothed CDFs.
    
    The source CDF is defined as:
        F_src(y) = sum_i w_src[i] * S(y - Y[i]),
    where
        S(z) = 0                  if z < -ε,
             = (z + ε)/(2ε)       if z in [-ε, ε],
             = 1                  if z > ε.
             
    The target quantile function Q_tgt is defined as the pseudo-inverse of
    F_tgt(x)= sum_i w_tgt[i] * S(x - Y[i]). In particular, if t is such that
        sum_{i=1}^j w_tgt[i] <= t <= sum_{i=1}^{j+1} w_tgt[i],
    then if we set
        s = (t - sum_{i=1}^j w_tgt[i]) / w_tgt[j+1],
    we define
        Q_tgt(t) = Y[j+1] - ε + 2ε·s,    if s > 0,
    and for s = 0 (i.e. t equals the cumulative sum up to j) we return
        Q_tgt(t) = Y[j] + ε.  (For j = 0 we simply use the standard formula.)
    
    Parameters:
        Y       : 1D torch.Tensor of sorted outcomes [Y₁, ..., Yₙ].
        w_src   : 1D torch.Tensor of weights for the source distribution.
        w_tgt   : 1D torch.Tensor of weights for the target distribution.
        epsilon : float, the window half-width.
    
    Returns:
        A function KR(y) that maps a torch.Tensor y to the transported value.
    """
    
    # Define the window function S(z)
    def S(z):
        return torch.where(z < -epsilon,
                           torch.zeros_like(z),
                           torch.where(z > epsilon,
                                       torch.ones_like(z),
                                       (z + epsilon) / (2 * epsilon)))
    
    # Define F_src(y) = ∑_i w_src[i] * S(y - Y[i])
    def F_src(y: torch.Tensor) -> torch.Tensor:
        # y can be a scalar or vector. We make it a column vector.
        if y.dim() == 1:
            y = y.unsqueeze(1)  # shape: (batch, 1)
        # Y is 1D of shape (n,); expand to (1, n)
        Y_exp = Y.unsqueeze(0)
        # Compute S(y - Y) elementwise and weight by w_src (of shape (n,))
        # Resulting shape: (batch, n)
        S_vals = S(y - Y_exp)
        # Multiply elementwise by w_src (broadcasted) and sum over outcomes.
        return torch.sum(w_src * S_vals, dim=-1)  # shape: (batch,)
    
    # Define Q_tgt(t): given t in [0,1], return x such that F_tgt(x)=t.
    def Q_tgt(t: torch.Tensor) -> torch.Tensor:
        # Ensure t is a 1D tensor.
        if t.dim() > 1:
            t = t.squeeze(-1)
        # Compute the cumulative sum of w_tgt.
        cumsum = torch.cumsum(w_tgt, dim=0)  # shape: (n,)
        # For each t, find the smallest index where cumsum exceeds t.
        # torch.searchsorted expects t and cumsum to be of compatible shapes.
        # We unsqueeze t to (batch, 1) and then squeeze the result.
        indices = torch.searchsorted(cumsum, t.unsqueeze(1)).squeeze(1)
        # Clamp indices to be between 0 and n-1.
        indices = torch.clamp(indices, 0, Y.numel() - 1)
        
        # Let j = indices - 1 (the last index for which cumulative sum is strictly below t),
        # and j_next = indices (the atom corresponding to the jump).
        j = torch.clamp(indices - 1, min=0)
        j_next = indices
        
        # Compute cumulative sum up to index j.
        cumsum_j = cumsum[j]  # shape: (batch,)
        # For each sample, get the weight corresponding to index j_next.
        theta_j_next = w_tgt[j_next]  # shape: (batch,)
        
        # Define s = (t - cumsum_j) / theta_j_next, which lies in [0,1].
        s = (t - cumsum_j) / theta_j_next
        
        # Retrieve Y[j_next] for each sample.
        Y_j_next = Y[j_next]
        # For cases where j > 0, also get Y[j-1].
        Y_prev = Y[torch.clamp(j - 1, min=0)]
        
        # For s > 0, set x = Y[j_next] - epsilon + 2ε·s.
        # For s == 0, to pick the infimum, set x = Y_prev + epsilon (if j > 0), else use the standard formula.
        x_candidate = Y_j_next - epsilon + 2 * epsilon * s
        x = torch.where((s == 0) & (j > 0),
                        Y_prev + epsilon,
                        x_candidate)
        return x
    
    def KR(y: torch.Tensor) -> torch.Tensor:
        """
        The modified KR transport is given by Q_tgt(F_src(y)).
        """
        t_val = F_src(y)
        return Q_tgt(t_val)
    
    return KR

# Example usage:
if __name__ == "__main__":
    n = 101
    # Create a sorted tensor of outcomes.
    Y = torch.linspace(0.0, 1.0, steps=n)
    # Example weights for two distributions (normalized).
    w_src = torch.ones(n) / n  # uniform weights for P_{Y|x1}
    # For illustration, use a slightly different target weight distribution.
    w_ind = torch.linspace(5, 1, steps=n)
    w_tgt = torch.linspace(1, 2, steps=n)
    w_ind = w_ind / w_ind.sum()
    w_tgt = w_tgt / w_tgt.sum()  # normalize
    epsilon = 1e-6
    KR_transport_chain1 = build_modified_KR_transport(Y, w_src, w_ind, epsilon)
    KR_transport_chain2 = build_modified_KR_transport(Y, w_ind, w_tgt, epsilon)
    KR_transport_direct = build_modified_KR_transport(Y, w_src, w_tgt, epsilon)
    
    # Evaluate the transport on some y values.
    y_input = torch.linspace(0.0, 1.0, steps=n-1)
    ind = KR_transport_chain1(y_input)
    transported_chain = KR_transport_chain2(ind)
    transported_direct = KR_transport_direct(y_input)
    print("Input y:", y_input)
    print("Transported y (chain):", transported_chain)
    print("Transported y (direct):", transported_direct)
