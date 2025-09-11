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

class NWConditioner(nn.Module):
    def __init__(self, X_train: torch.Tensor, kernel):
        """
        NWConditioner computes weights using a Nadaraya–Watson kernel regression.
        
        Parameters:
          X_train : torch.Tensor
              A 2D tensor of training inputs of shape (n, d).
          kernel : a kernel object
              For example, an instance of GaussianKernel that has a 'lengthscale' attribute.
        """
        super().__init__()
        # Store X_train as a buffer so it moves with the model
        self.register_buffer("X_train", X_train)
        self.kernel = kernel
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        For each input in X (of shape (M, d)), compute the kernel similarity
        to each training point. Return a tensor of shape (M, n) that is normalized along dim=1.
        """
        # Compute pairwise distances: shape (M, n)
        # Here we assume self.kernel.get_gram computes the kernel Gram matrix.
        K = self.kernel.get_gram(X, self.X_train)
        # Normalize each row to sum to 1:
        theta = K / (K.sum(dim=1, keepdim=True) + 1e-8)
        return theta


import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RFConditioner(nn.Module):
    def __init__(self, rf_estimator: RandomForestRegressor, X_train: np.ndarray, Y_train: np.ndarray):
        """
        A Random Forest Conditioner that outputs normalized weights (theta) over the training outcomes,
        following a QR forest procedure.
        
        Parameters:
            rf_estimator: Trained scikit-learn RandomForestRegressor acting as a QR forest.
            X_train: Training features as a NumPy array of shape (n_train, d).
            Y_train: Training outcomes as a 1D NumPy array of length n_train.
        
        The conditioner precomputes, for each tree, a mapping from each leaf index
        to the list of training indices that fall into that leaf.
        """
        super().__init__()
        self.rf = rf_estimator
        self.X_train = X_train  # kept as NumPy array for scikit-learn calls
        self.Y_train = Y_train  # may be used for debugging or further processing
        self.n_train = X_train.shape[0]
        
        # Pre-compute per-tree leaf mappings: for each tree, record a dictionary
        # mapping leaf id -> array of training indices in that leaf.
        self.leaf_maps = []
        for tree in self.rf.estimators_:
            leaf_ids = tree.apply(self.X_train)  # shape (n_train,)
            leaf_map = {}
            for i, leaf in enumerate(leaf_ids):
                if leaf not in leaf_map:
                    leaf_map[leaf] = []
                leaf_map[leaf].append(i)
            # Convert lists to numpy arrays for convenience
            for leaf in leaf_map:
                leaf_map[leaf] = np.array(leaf_map[leaf])
            self.leaf_maps.append(leaf_map)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes, for each query input in X (shape M x d), a weight vector over the training samples.
        
        For each query sample, for each tree in the forest, we determine the leaf node in which the sample falls.
        Then, we assign a unit count to all training points that reside in that same leaf.
        Finally, we average over trees and normalize the resulting weight vector so that it sums to 1.
        
        Returns:
            A torch.Tensor of shape (M, n_train) where each row sums to 1.
        """
        # Convert X to NumPy for scikit-learn compatibility.
        X_np = X.detach().cpu().numpy()
        M = X_np.shape[0]
        weights = np.zeros((M, self.n_train))
        
        # For each tree, determine leaf assignment for each query sample and accumulate counts.
        for leaf_map, tree in zip(self.leaf_maps, self.rf.estimators_):
            leaf_ids = tree.apply(X_np)  # shape (M,)
            for i, leaf in enumerate(leaf_ids):
                if leaf in leaf_map:
                    indices = leaf_map[leaf]
                    weights[i, indices] += 1
        # Average over the number of trees
        weights /= len(self.rf.estimators_)
        # Normalize each row to sum to 1.
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        # Convert back to a torch.Tensor
        return torch.tensor(weights, dtype=torch.float32, device=X.device)


class AggregateConditioner(torch.nn.Module):
    def __init__(self, conditioners, domain_key="D", feature_key="X"):
        super().__init__()
        self.conditioners = torch.nn.ModuleList(conditioners)
        self.domain_key = domain_key
        self.feature_key = feature_key

    def forward(self, input_dict):
        X_query = input_dict[self.feature_key]  # shape (M, d)
        D_query = input_dict[self.domain_key]   # shape (M,)
        
        M = X_query.shape[0]
        # Assume all conditioners return same output dim — grab dim from first output
        with torch.no_grad():
            sample_output = self.conditioners[0](X_query[:1])
        output_dim = sample_output.shape[1]
        
        theta = torch.zeros(M, output_dim, device=X_query.device)
        for d in torch.unique(D_query):
            d_int = d.item()
            mask = (D_query == d)
            theta[mask] = self.conditioners[d_int](X_query[mask])
        return theta


pi = torch.acos(torch.zeros(1)).item() * 2