import torch
import torch.nn as nn
from causal_cocycle.RQS import unconstrained_RQS
import torch.nn.functional as F

class Scale_layer(nn.Module):
    """
    g(theta, y) -> softplus(theta) * y, using a numerically‑stable softplus
    """
    def __init__(self):
        super().__init__()
        # F.softplus clamps internally, equivalent to log(1+exp(x)) but stable
        self.transform = F.softplus

    def forward(self, theta, y, logdet=False):
        scale = self.transform(theta)
        if logdet:
            ld = torch.log(scale.squeeze(-1))
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
    def __init__(self, transform=F.softplus):
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
    def __init__(self, bins=8, min_width=1e-3, min_height=1e-3, min_derivative=1e-3, tail_bound = 3.0):
        super().__init__()
        self.min_width = min_width
        self.min_height = min_height
        self.min_derivative = min_derivative
        self.bins = bins
        self.tail_bound = tail_bound
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
            tail_bound=self.tail_bound,
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
            tail_bound=self.tail_bound,
            min_bin_width=self.min_width,
            min_bin_height=self.min_height,
            min_derivative=self.min_derivative,
            log_det=logdet
        )
        self.inputs_in_mask_itercount += in_mask_count
        return u.view(len(y), 1), ld

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
    we modify the pseudo-inverse so that if s == 0, we would have returned the appropriate boundary.
    (Note: in our implementation with distinct Y_train values, s==0 is not expected.)
    """
    
    def __init__(self, Y_train: torch.Tensor, epsilon: float):
        """
        Parameters:
            Y_train : torch.Tensor
                A 1D tensor of SORTED UNIQUE training outcomes [Y_1, ..., Y_n].
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
        # Convert z to double precision for sensitivity.
        z = z.double()
        # Use self.epsilon rather than a global epsilon.
        if self.epsilon == 0:
            return (z >= 0).float()
        else:
            return torch.where(z < -self.epsilon, torch.zeros_like(z),
                               torch.where(z > self.epsilon, torch.ones_like(z),
                                           (z + self.epsilon) / (2 * self.epsilon)))

    def forward(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward mapping Q: [0,1] -> Y-space (the pseudo-inverse of F)
        for a batch of weight vectors.
        
        Parameters:
            theta : torch.Tensor
                Tensor of shape (batch_size, n) representing normalized weights.
            t : torch.Tensor
                Tensor of target cumulative probabilities in [0,1] of shape (batch_size,)
                or (batch_size, 1).
        
        Returns:
            torch.Tensor of shape (batch_size,) containing the corresponding x values.
        """
        # Ensure t is a tensor on the right device.
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.Y_train.device)
        # If t has an extra singleton dimension, squeeze it.
        if t.dim() > 1:
            t = t.reshape(-1)  # now t has shape (batch,)
            
        batch_size, n = theta.shape
        
        # Compute cumulative sum for each batch element along the weights dimension.
        cumsum = torch.cumsum(theta, dim=1)  # shape: (batch, n)
        
        # Vectorized searchsorted:
        indices = torch.searchsorted(cumsum, t.unsqueeze(1)).squeeze(1)
        indices = torch.clamp(indices, 0, self.Y_train.numel()-1)
        
        # j: lower index for each sample.
        j = indices - 1  # shape: (batch,)
        j_next = indices  # shape: (batch,)
        
        # Prepend zero to each cumulative sum row.
        zero_vec = torch.zeros(batch_size, 1, device=theta.device, dtype=theta.dtype)
        cumsum_ext = torch.cat((zero_vec, cumsum), dim=1)  # shape: (batch, n+1)
        
        # For each batch element, select cumsum_ext at index (j+1)
        # (Because we prepended a zero, index j+1 corresponds to original cumsum at index j.)
        cumsum_j = cumsum_ext[torch.arange(batch_size), (j + 1).long()]  # shape: (batch,)
        
        # Retrieve w_{j_next} from theta.
        w_j_next = theta[torch.arange(batch_size), j_next.long()]  # shape: (batch,)
        
        # Compute interpolation factor s.
        s = (t - cumsum_j) / w_j_next  # shape: (batch,)
        
        # Retrieve the corresponding Y value using batched indexing.
        Y_j_next = self.Y_train[j_next.long()].double()  # shape: (batch,)
        
        # Compute candidate inverse value.
        result = Y_j_next - self.epsilon + 2 * self.epsilon * s  # shape: (batch,)
        return result

    def backward(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vectorized backward mapping (CDF): Given y, compute
           F(y) = sum_{j=1}^n theta_j S(y - Y_j)
        for a batch of y values.
        
        Parameters:
            theta : torch.Tensor
                Tensor of shape (batch_size, n) representing normalized weights.
            y : torch.Tensor
                Tensor of shape (batch_size,) or (batch_size, 1) representing values in Y-space.
        
        Returns:
            torch.Tensor of shape (batch_size, 1) containing F(y).
        """
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.Y_train.device)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(1)  # Now shape: (batch, 1)
            
        # Expand Y_train to shape (1, n) to broadcast against y.
        Y_exp = self.Y_train.unsqueeze(0)  # shape: (1, n)
        # Compute the difference (y - Y_train) for each batch element.
        diff = y - Y_exp  # shape: (batch, n)
        # Compute S(diff) elementwise.
        S_vals = self.S(diff)  # shape: (batch, n)
        # Compute weighted sum for each batch element.
        F_vals = torch.sum(theta * S_vals, dim=1)  # shape: (batch,)
        return F_vals.unsqueeze(1)  # Shape: (batch, 1)

# Example usage:
if __name__ == "__main__":
    # Sorted training outcomes.
    Y_train = torch.linspace(0.0, 1.0, steps=101)
    epsilon = 1e-4
    transformer = KREpsLayer(Y_train, epsilon)
    
    # For a batch of examples, use uniform weights.
    batch_size = 5
    n_train = Y_train.numel()
    theta = torch.randn((batch_size, n_train)).abs()
    theta = theta / theta.sum(1)[:,None]  # normalized weights
    
    # Forward mapping: given target t values in [0,1], recover x.
    t_input = torch.tensor([[0.0, 0.1, 0.35, 0.75, 0.95]]).T  # note t=0 is included.
    x_output = transformer.forward(theta, t_input)
    print("Forward (quantile) output:", x_output)
    
    # Backward mapping: given x, compute F(x).
    F_x = transformer.backward(theta, x_output)
    print("Backward (CDF) output:", F_x, "original :", t_input)


