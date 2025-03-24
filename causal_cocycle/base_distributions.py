# base_distributions.py
import torch
import torch.nn as nn
from torch.distributions import Normal, StudentT, Laplace

class ParameterizedNormal(nn.Module):
    """
    A parameterized Normal distribution with learnable mean and standard deviation.
    """
    def __init__(self, dim=1, init_mean=0.0, init_std=1.0):
        super().__init__()
        # Register learnable parameters.
        self.mean = nn.Parameter(torch.full((dim,), init_mean))
        # We store log_std for numerical stability.
        self.log_std = nn.Parameter(torch.full((dim,), torch.log(torch.tensor(init_std, dtype=torch.float))))
        
    def forward(self):
        std = torch.exp(self.log_std)
        return Normal(self.mean, std)

class ParameterizedStudentT(nn.Module):
    """
    A parameterized StudentT distribution with learnable location, scale, and degrees of freedom.
    """
    def __init__(self, dim=1, init_loc=0.0, init_scale=1.0, init_df=5.0):
        super().__init__()
        self.loc = nn.Parameter(torch.full((dim,), init_loc))
        self.log_scale = nn.Parameter(torch.full((dim,), torch.log(torch.tensor(init_scale, dtype=torch.float))))
        # Parameterize degrees of freedom in log-space to ensure positivity.
        self.log_df = nn.Parameter(torch.full((dim,), torch.log(torch.tensor(init_df, dtype=torch.float))))
        
    def forward(self):
        scale = torch.exp(self.log_scale)
        df = torch.exp(self.log_df)
        return StudentT(df, self.loc, scale)

class ParameterizedLaplace(nn.Module):
    """
    A parameterized Laplace distribution with learnable location and scale.
    """
    def __init__(self, dim=1, init_loc=0.0, init_scale=1.0):
        super().__init__()
        self.loc = nn.Parameter(torch.full((dim,), init_loc))
        self.log_scale = nn.Parameter(torch.full((dim,), torch.log(torch.tensor(init_scale, dtype=torch.float))))
        
    def forward(self):
        scale = torch.exp(self.log_scale)
        return Laplace(self.loc, scale)
