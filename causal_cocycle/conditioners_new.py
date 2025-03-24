import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform
import math

class CompositeConditioner(nn.Module):
    def __init__(self, layers):
        """
        Combines multiple conditioners.
        conditioners: list of conditioner modules.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
     
    def forward(self, X):
        outputs = []
        for cond in self.layers:
            outputs.append(cond(X))
        return outputs

class EmptyConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.ones(1))
    
    def forward(self, X):
        return torch.zeros((len(X), 1), device=X.device)

class ConstantConditioner_1D(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([init], dtype=torch.float32))
    
    def forward(self, X):
        return self.param.expand(len(X), 1)

class ConstantConditioner(nn.Module):
    def __init__(self, init=1.0, full=True, grad=True):
        super().__init__()
        if not isinstance(init, torch.Tensor):
            init = torch.tensor([init], dtype=torch.float32)
        self.param = nn.Parameter(init, requires_grad=grad)
        self.full = full
    
    def forward(self, X):
        if self.full:
            # Broadcast the constant to match batch size.
            return torch.ones((len(X), 1), device=X.device) @ self.param
        else:
            return self.param

class LinConditioner(nn.Module):
    def __init__(self, d, p, bias=True, init=None, grad=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(d, p, bias=bias)
        if init is not None:
            with torch.no_grad():
                self.linear.weight.fill_(init)
            self.linear.weight.requires_grad = grad
    
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

class RFFConditioner(nn.Module):
    def __init__(self, features, input_dims, sd=1):
        super().__init__()
        self.features = features
        self.input_dims = input_dims
        self.weights = nn.Parameter(Normal(0, sd/features).sample((features, 1)))
        self.feature_scale = nn.Parameter(Normal(0, sd/input_dims).sample((features, input_dims)))
        self.feature_shift = nn.Parameter(Uniform(0, 2*math.pi).sample((features, 1)))
        self.input_dim_relevance = nn.Parameter(torch.ones((input_dims,)))
    
    def forward(self, X):
        scaled_X = X * self.input_dim_relevance
        return (self.weights.T @ torch.cos(self.feature_scale @ scaled_X.T + self.feature_shift)).T

class NNConditioner(nn.Module):
    def __init__(self, width, layers, input_dims, output_dims=1, bias=True, activation=nn.ReLU()):
        """
        Constructs a fully connected network for conditioning.

        Parameters:
        -----------
        width : int
            The number of hidden units in each hidden layer.
        layers : int
            The total number of layers (including the first layer from input_dims to width,
            and hidden layers).
        input_dims : int
            The dimensionality of the input.
        output_dims : int, default 1
            The dimensionality of the output.
        bias : bool, default True
            Whether to include a bias term in the linear layers.
        activation : nn.Module, default nn.ReLU()
            The activation function to use (e.g. nn.ReLU(), nn.ELU(), nn.Tanh(), etc.).
        """
        super().__init__()
        self.flatten = nn.Flatten()
        modules = []
        # First layer: from input_dims to width.
        modules.append(nn.Linear(input_dims, width, bias=bias))
        modules.append(activation)
        # Hidden layers: we already used one layer, so add (layers-1) more hidden layers.
        for _ in range(layers - 1):
            modules.append(nn.Linear(width, width, bias=bias))
            modules.append(activation)
        # Output layer: from width to output_dims.
        modules.append(nn.Linear(width, output_dims, bias=bias))
        self.net = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

pi = torch.acos(torch.zeros(1)).item() * 2