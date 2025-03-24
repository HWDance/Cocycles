import torch
import torch.nn as nn
from .RQS import unconstrained_RQS

def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)

class Transformer(nn.Module):
    """
    Aggregate transformer that takes in monotone map layers.
    If logdet=True, the forward/backward calls accumulate log|det(Jacobian)|.
    """
    def __init__(self, layers, logdet=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.logdet = logdet
        
    def forward(self, theta, y):
        """
        Forward pass: Applies each layer in reverse order, collecting logdet if self.logdet=True.
        """
        logdet_sum = torch.zeros(len(y), device=y.device)
        out = y
        for i in range(len(self.layers)):
            layer = len(self.layers)-(i+1)
            out, ld = self.layers[layer].forward(theta[layer], out, self.logdet)
            logdet_sum += ld
        if self.logdet:
            return out, logdet_sum
        else:
            return out
        
    def backward(self, theta, y):
        """
        Inverse pass: Applies each layer in forward order, collecting logdet if self.logdet=True.
        """
        logdet_sum = torch.zeros(len(y), device=y.device)
        out = y
        for i in range(len(self.layers)):
            out, ld = self.layers[i].backward(theta[i], out, self.logdet)
            logdet_sum += ld
        if self.logdet:
            return out, logdet_sum
        else:
            return out

class ShiftLayer(nn.Module):
    """
    g(theta, y) -> transform(theta) + y
    """
    def __init__(self, transform=lambda x: x):
        super().__init__()
        self.transform = transform
    
    def forward(self, theta, y, logdet=False):
        ld = torch.zeros(len(y), device=y.device) if logdet else 0
        return self.transform(theta) + y, ld
    
    def backward(self, theta, y, logdet=False):
        ld = torch.zeros(len(y), device=y.device) if logdet else 0
        return y - self.transform(theta), ld
    
class ScaleLayer(nn.Module):
    """
    g(theta, y) -> transform(theta) * y
    By default transform = log(1+exp(...)) for positivity
    """
    def __init__(self, transform=lambda x: torch.log(1 + torch.exp(x))):
        super().__init__()
        self.transform = transform
        
    def forward(self, theta, y, logdet=False):
        scale = self.transform(theta)
        if logdet:
            # log|det| = log(scale)
            ld = torch.log(scale.squeeze(-1))  # for 1D
        else:
            ld = 0
        return scale * y, ld
    
    def backward(self, theta, y, logdet=False):
        scale = self.transform(theta)
        if logdet:
            ld = -torch.log(scale.squeeze(-1))
        else:
            ld = 0
        return y / scale, ld
    
class InverseLayer(nn.Module):
    """
    g(theta, y) -> theta / y
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, theta, y, logdet=False):
        out = theta / y
        if logdet:
            # derivative wrt y is -theta / y^2 => partial log determinant logic
            ld = (torch.log(theta) - 2 * torch.log(y)).view(-1)
        else:
            ld = 0
        return out, ld
    
    def backward(self, theta, y, logdet=False):
        out = theta / y
        if logdet:
            ld = (torch.log(theta) - 2 * torch.log(y)).view(-1)
        else:
            ld = 0
        return out, ld

class HyperbolicRELULayer(nn.Module):
    """
    g(theta, y) -> piecewise function for y >= knot or y < knot.
    'theta' is unused here (no direct param?), so it's just
    a placeholder for consistency.
    """
    def __init__(self, knot, domain_flip=False):
        super().__init__()
        self.knot = knot
        self.flip = domain_flip
    
    def forward(self, theta, y, logdet=False):
        # piecewise
        f1 = y
        f2 = (self.knot + 1/self.knot - 1/y)
        out = (-1)**self.flip * (f1 * (y >= self.knot) + f2 * (y < self.knot))

        if logdet:
            ld = (1 * (y >= self.knot) + (1 / y**2) * (y < self.knot)).view(-1)
        else:
            ld = 0
        return out, ld
    
    def backward(self, theta, y, logdet=False):
        # invert piecewise
        y_in = ((-1)**self.flip) * y
        f1inv = y_in
        f2inv = 1 / (self.knot + 1/self.knot - y_in)
        out = f1inv * (y_in >= self.knot) + (-1)**self.flip * f2inv * (y_in < self.knot)

        if logdet:
            ld = (1 * (y_in >= self.knot) + (f2inv**2) * (y_in < self.knot)).view(-1)
        else:
            ld = 0
        return out, ld

class RQSLayer(nn.Module):
    """
    g(theta, y) -> RQS spline transformation
    Uses unconstrained_RQS from RQS.py
    """
    def __init__(self, bins=8, min_width=1e-3, min_height=1e-3, min_derivative=1e-3):
        super().__init__()
        self.min_width = min_width
        self.min_height = min_height
        self.min_derivative = min_derivative
        self.bins = bins
        self.inputs_in_mask_itercount = 0

    def forward(self, theta, y, logdet=False):
        """
        forward pass for RQS (from your original code).
        """
        Tu, ld, in_mask_count = unconstrained_RQS(
            y.view(len(y),),
            theta[:, :self.bins],
            theta[:, self.bins:2*self.bins],
            theta[:, 2*self.bins:(3*self.bins+1)],
            inverse=False,
            tail_bound=theta[:, -1].abs().mean(),
            min_bin_width=self.min_width,
            min_bin_height=self.min_height,
            min_derivative=self.min_derivative,
            log_det=logdet
        )
        self.inputs_in_mask_itercount += in_mask_count
        return Tu.view(len(y), 1), ld
    
    def backward(self, theta, y, logdet=False):
        """
        inverse pass for RQS
        """
        u, ld, in_mask_count = unconstrained_RQS(
            y.view(len(y),),
            theta[:, :self.bins],
            theta[:, self.bins:2*self.bins],
            theta[:, 2*self.bins:(3*self.bins+1)],
            inverse=True,
            tail_bound=theta[:, -1].abs().mean(),
            min_bin_width=self.min_width,
            min_bin_height=self.min_height,
            min_derivative=self.min_derivative,
            log_det=logdet
        )
        self.inputs_in_mask_itercount += in_mask_count
        return u.view(len(y), 1), ld

import torch
import torch.nn as nn

class KRLayer(nn.Module):
    """
    KR Transport Transformer using the raw (hard-threshold) KR mapping.
    
    Assumptions:
      • Y_train is a 1D tensor of sorted outcomes: [Y_1, ..., Y_n].
      • The conditioner outputs a normalized weight vector theta (with sum(theta)=1).
    
    Backward mapping (CDF):
        F(y) = ∑_{i=1}^n θ_i * 1{Y_i ≤ y}
    
    Forward mapping (quantile function):
        Given t in [0,1], compute the cumulative sum of θ and return Y_i such that
        the cumulative sum up to i-1 is below t and the cumulative sum up to i is above t.
    """
    
    def __init__(self, Y_train: torch.Tensor):
        super().__init__()
        if Y_train.dim() != 1:
            raise ValueError("Y_train must be a 1D tensor of sorted outcomes.")
        self.register_buffer("Y_train", Y_train)
    
    def forward(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward mapping Q: [0,1] -> Y-space.
        
        Parameters:
          theta : torch.Tensor
              A tensor of shape (batch_size, n) representing normalized weights.
          t : torch.Tensor
              A tensor of input quantile values in [0,1] of shape (batch_size,) or (batch_size, 1).
        
        Returns:
          torch.Tensor
              A tensor of quantile values from Y_train corresponding to each t.
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # Compute cumulative sum along the weight dimension.
        cumsum = torch.cumsum(theta, dim=-1)  # shape: (batch_size, n)
        # Find the smallest index where the cumulative sum exceeds t.
        indices = torch.searchsorted(cumsum, t)
        indices = torch.clamp(indices, 0, self.Y_train.numel() - 1)
        return self.Y_train[indices.squeeze(-1)]
    
    def backward(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Backward mapping F: Y -> [0,1] (the conditional CDF).
        
        Parameters:
          theta : torch.Tensor
              A tensor of shape (batch_size, n) representing normalized weights.
          y : torch.Tensor
              A tensor of input values in Y-space of shape (batch_size,) or (batch_size, 1).
        
        Returns:
          torch.Tensor
              A tensor of cumulative probabilities F(y) of shape (batch_size, 1).
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
        # Broadcast Y_train to compare with each y.
        Y_train_exp = self.Y_train.unsqueeze(0)  # shape: (1, n)
        # Hard indicator: 1 if Y_train <= y, 0 otherwise.
        indicator = (Y_train_exp <= y).float()
        F_val = torch.sum(theta * indicator, dim=-1, keepdim=True)
        return F_val

class KREpsLayer(nn.Module):
    """
    Smoothed KR Transport Transformer using the notation S_j(x)=S(x-Y_j),
    where
      S(z) = 0              if z < -eps,
           = (z+eps)/(2eps)  if z in [-eps, eps],
           = 1              if z > eps.
    
    The smoothed CDF is defined as
        F(x) = sum_{j=1}^n theta_j S_j(x),
    with ∑_j theta_j = 1.
    
    For inversion (the forward mapping Q), given a target t in [0,1] we first find the index j
    so that
         ∑_{i=1}^j theta_i ≤ t ≤ ∑_{i=1}^{j+1} theta_i.
    
    Then we have:
         t = ∑_{i=1}^j theta_i + theta_{j+1} S_{j+1}(x).
    
    Setting s = S_{j+1}(x) = (t - ∑_{i=1}^j theta_i) / theta_{j+1},
    the standard pseudo-inverse on the linear part would be:
         x = Y_{j+1} - eps + 2eps s.
    
    For consistency with the generalized inverse (taking the infimum x with F(x) ≥ t),
    we modify the pseudo-inverse so that if s == 0 (i.e. t exactly equals the cumulative sum up to j),
    we return x = Y_{j-1} + eps.
    
    For s > 0 we set:
         x = eps (2s - 1) + Y_{j+1}.
    """
    
    def __init__(self, Y_train: torch.Tensor, epsilon: float):
        """
        Parameters:
            Y_train : torch.Tensor
                A 1D tensor of sorted training outcomes [Y_1, ..., Y_n].
            epsilon : float
                The window half-width.
        """
        super().__init__()
        if Y_train.dim() != 1:
            raise ValueError("Y_train must be a 1D tensor of sorted outcomes.")
        self.register_buffer("Y_train", Y_train)
        self.epsilon = epsilon

    def S(self, z: torch.Tensor) -> torch.Tensor:
        """
        Window function S(z)= (z+eps)/(2eps) for z in [-eps, eps],
        0 for z < -eps and 1 for z > eps.
        """
        return torch.where(z < -self.epsilon,
                           torch.zeros_like(z),
                           torch.where(z > self.epsilon,
                                       torch.ones_like(z),
                                       (z + self.epsilon) / (2 * self.epsilon)))
    
    def forward(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward mapping Q: [0,1] -> Y-space (i.e. the pseudo-inverse of F).
        
        Parameters:
            theta : torch.Tensor
                Tensor of shape (batch_size, n) representing normalized weights.
            t : torch.Tensor
                Tensor of target cumulative probabilities in [0,1] of shape (batch_size,)
                or (batch_size, 1).
        
        Returns:
            torch.Tensor of shape (batch_size,) containing the corresponding x values.
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Compute cumulative weights (batch_size, n).
        cumsum = torch.cumsum(theta, dim=-1)
        # For each sample, find the smallest index where cumsum exceeds t.
        indices = torch.searchsorted(cumsum, t).squeeze(-1)
        indices = torch.clamp(indices, 0, self.Y_train.numel() - 1)
        
        # Let j be indices - 1 (cumulative sum strictly below t), and j_next = indices.
        j = torch.clamp(indices - 1, min=0)
        j_next = indices
        
        # Cumulative sum up to index j.
        cumsum_j = torch.gather(cumsum, 1, j.unsqueeze(1))
        # Weight at index j_next.
        theta_j_next = torch.gather(theta, 1, j_next.unsqueeze(1))
        # Define s = S_{j+1}(x) = (t - cumsum_j) / theta_{j+1}.
        s = (t - cumsum_j) / theta_j_next  # s in [0,1]
        
        # For samples where s == 0, we return the infimum x, i.e. Y_{j-1} + eps.
        # For s > 0, we return: x = Y_{j+1} - eps + 2eps s = eps(2s - 1) + Y_{j+1}.
        # Handle the edge case where j==0 (no j-1 exists) by simply using the standard formula.
        Y_prev = self.Y_train[torch.clamp(j - 1, min=0)]
        x_candidate = self.Y_train[j_next] - self.epsilon + 2 * self.epsilon * s.squeeze(1)
        
        x = torch.where(s.squeeze(1) == 0,
                        torch.where(j > 0, Y_prev + self.epsilon, x_candidate),
                        x_candidate)
        return x

    def backward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Backward mapping (CDF): Given x, compute
            F(x) = sum_{j=1}^n theta_j S_j(x),
        where S_j(x)=S(x-Y_j).
        
        Parameters:
            theta : torch.Tensor
                Tensor of shape (batch_size, n) representing normalized weights.
            x : torch.Tensor
                Tensor of shape (batch_size,) or (batch_size, 1) representing values in Y-space.
        
        Returns:
            torch.Tensor of shape (batch_size, 1) containing F(x).
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        Y_train_exp = self.Y_train.unsqueeze(0)  # shape (1, n)
        z = x - Y_train_exp  # shape (batch_size, n)
        S_vals = self.S(z)
        F_val = torch.sum(theta * S_vals, dim=-1, keepdim=True)
        return F_val

# Example usage:
if __name__ == "__main__":
    # Sorted training outcomes.
    Y_train = torch.linspace(0.0, 1.0, steps=100)
    epsilon = 0.05
    transformer = KRLayerEps(Y_train, epsilon)
    
    # For a batch of examples, use uniform weights.
    batch_size = 5
    n_train = Y_train.numel()
    theta = torch.ones((batch_size, n_train)) / n_train  # normalized weights
    
    # Forward mapping: given target t values in [0,1], recover x.
    t_input = torch.tensor([0.0, 0.1, 0.35, 0.75, 0.95])  # note t=0 is included.
    x_output = transformer.forward(theta, t_input)
    print("Forward (quantile) output:", x_output)
    
    # Backward mapping: given x, compute F(x).
    F_x = transformer.backward(theta, x_output)
    print("Backward (CDF) output:", F_x.squeeze())


