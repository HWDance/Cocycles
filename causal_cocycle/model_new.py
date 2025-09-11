import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from zuko.flows.autoregressive import MaskedAutoregressiveTransform

class ZukoCocycleModel(nn.Module):
    def __init__(self, transforms: nn.ModuleList):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def transformation(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Generation: latent u → y under context x.
        """
        y = u
        # build the conditional transform once
        for T in self.transforms:
            transform = T(x)        # T.forward(x) → AutoregressiveTransform
            y = transform.inv(y)
        return y

    def inverse_transformation(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Abduction: observed y → u under context x.
        """
        u = y
        for T in reversed(self.transforms):
            transform = T(x)
            u = transform(u)
        return u

    def cocycle(self, x1: Tensor, x2: Tensor, y: Tensor) -> Tensor:
        u = self.inverse_transformation(x2, y)
        return self.transformation(x1, u)

    def cocycle_outer(
        self,
        x1: Tensor,    # (M, x_dim)
        x2: Tensor,    # (N, x_dim)
        y: Tensor      # (N, y_dim)
    ) -> Tensor:      # → (M, N, y_dim)
        M, N = x1.size(0), y.size(0)

        # 1) Abduct each y[j] → u[j]
        u = self.inverse_transformation(x2, y)  # (N, y_dim)

        # 2) Build all (i,j) pairs
        x1_rep = x1.unsqueeze(1).expand(M, N, x1.size(1)).reshape(M*N, -1)
        u_rep  = u.unsqueeze(0).expand(M, N, u.size(1)).reshape(M*N, -1)

        # 3) Generate in one shot
        v_rep = self.transformation(x1_rep, u_rep)  # (M*N, y_dim)

        # 4) Reshape back to grid
        return v_rep.view(M, N, -1)


class CocycleOutcomeModel(nn.Module):
    def __init__(self, model, inputs_train, outputs_train):
        """
        model: cocycle model (an nn.Module)
        inputs_train: training inputs (N x D torch.Tensor)
        outputs_train: training outputs (N x P torch.Tensor)
        """
        super().__init__()
        self.model = model
        self.inputs = inputs_train
        self.outputs = outputs_train

    def forward(self, inputs, feature):
        prediction = self.model.cocycle_outer(inputs, self.inputs, self.outputs)  # M x N x P
        return feature(prediction).mean(2)

class FlowOutcomeModel(nn.Module):
    def __init__(self, model, noise_samples):
        """
        model: flow model (an nn.Module)
        noise_samples: noise samples from base distribution (N x P torch.Tensor)
        """
        super().__init__()
        self.model = model
        self.noise = noise_samples

    def forward(self, inputs, feature):
        prediction = self.model.transformation_outer(inputs, self.noise)  # M x N x P
        return feature(prediction).mean(2)

class CocycleModel(nn.Module):
    def __init__(self, conditioner, transformer):
        """
        conditioner: a composite conditioner module that returns a list of outputs
                     (an instance of Compositeconditioner)
        transformer: an invertible transformer module
        """
        super().__init__()
        # Expect conditioner to be a composite module.
        self.conditioner = conditioner
        self.transformer = transformer

    def transformation(self, x, y):
        transformer_parameters = self.conditioner(x)  # Returns a list of outputs
        return self.transformer.forward(transformer_parameters, y)

    def transformation_outer(self, x, y):
        transformer_parameters = []
        M = x['X'].shape[0] if isinstance(x, dict) else x.shape[0]
        N = y.shape[0]
    
        eye_y = torch.ones((N, 1), device=y.device)
        eye_x = torch.ones((M, 1), device=y.device)
    
        for cond in self.conditioner.layers:
            theta = cond(x)  # (M, p)
            outer_parameters = torch.kron(theta, eye_y)  # (M*N, p)
            outer_y = torch.kron(eye_x, y)              # (M*N, q)
            transformer_parameters.append(outer_parameters)
    
        return self.transformer.forward(transformer_parameters, outer_y).reshape(M, N)


    def inverse_transformation(self, x, y):
        transformer_parameters = self.conditioner(x)
        return self.transformer.backward(transformer_parameters, y)

    def cocycle(self, x1, x2, y):
        return self.transformation(x1, self.inverse_transformation(x2, y))

    def cocycle_outer(self, x1, x2, y):
        return self.transformation_outer(x1, self.inverse_transformation(x2, y))

class FlowModel(nn.Module):
    """
    Similar to CocycleModel but structured to support log-determinants.
    """
    def __init__(self, conditioner, transformer, base_dist):
        super().__init__()
        self.conditioner = conditioner
        self.transformer = transformer
        self.base_distribution_module = base_dist  # Register as an attribute

    @property
    def base_distribution(self):
        # When you need the distribution, call forward on the module.
        return self.base_distribution_module()

    def transformation(self, x, y):
        transformer_parameters = self.conditioner(x)
        return self.transformer.forward(transformer_parameters, y)

    def transformation_outer(self, x, y):
        transformer_parameters = []
        for cond in self.conditioner.layers:
            eye = torch.ones((len(y), 1), device=y.device)
            outer_parameters = torch.kron(cond(x), eye)
            outer_y = torch.kron(eye, y)
            transformer_parameters.append(outer_parameters)
        return self.transformer.forward(transformer_parameters, outer_y).reshape(len(x), len(y))

    def inverse_transformation(self, x, y):
        transformer_parameters = self.conditioner(x)
        return self.transformer.backward(transformer_parameters, y)

    def cocycle(self, x1, x2, y):
        return self.transformation(x1, self.inverse_transformation(x2, y))

    def cocycle_outer(self, x1, x2, y):
        return self.transformation_outer(x1, self.inverse_transformation(x2, y))

class ZukoFlowModel(nn.Module):
    """
    A conditional normalizing flow model with explicit base distribution.

    Attributes:
        transforms (nn.ModuleList): sequence of lazy bijectors T, each called as T(x)
            to produce a Transform object whose .forward/.inverse operate on data.
        base_distribution (torch.distributions.Distribution):
            latent prior p(u), e.g. Normal or Laplace
    """
    def __init__(self,
                 transforms: nn.ModuleList,
                 base_distribution: Distribution):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.base_distribution = base_distribution

    def transformation(self,
                       x: torch.Tensor,
                       u: torch.Tensor) -> torch.Tensor:
        """
        Generation: latent u -> y under context x:
            y = T_L^{-1} ∘ … ∘ T_1^{-1}(u; context=x)
        """
        y = u
        for T in self.transforms:
            # freeze in the context to get a concrete Transform
            transform = T(x)           # equivalent to T.forward(x)
            y      = transform.inv(y)  # latent -> data
        return y

    def inverse_transformation(self,
                               x: torch.Tensor,
                               y: torch.Tensor) -> torch.Tensor:
        """
        Abduction: observed y -> u under context x:
            u = T_1 ∘ … ∘ T_L(y; context=x)
        """
        u = y
        # reversed() iterates through self.transforms in reverse order
        for T in reversed(self.transforms):
            transform = T(x)
            u      = transform(u)   # data -> latent
        return u

    def log_prob(self,
                 x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        """
        Compute log density log p(y | x) by
        1) abduction (data->latent) accumulating ldj
        2) base log-prob at u
        Returns a tensor of shape (batch_size,)
        """
        u = y
        total_ldj = torch.zeros(u.size(0), device=u.device)
        # 1) abduction: through each T in reverse
        for T in reversed(self.transforms):
            transform = T(x)
            u    = transform(u)
            ldj = transform.log_abs_det_jacobian(y,u)
            total_ldj = total_ldj + ldj
        # 2) base log prob
        #    .log_prob(u) gives shape (batch, y_dim), so sum over last axis
        logpu = self.base_distribution.log_prob(u).sum(dim=-1)
        return logpu + total_ldj

    def sample(self,
               x: torch.Tensor,
               num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from p(y|x) by sampling u ~ p(u) and generating y.
        Returns a tensor of shape (num_samples, batch_size, y_dim)
        """
        # 1) draw latent: (num_samples, batch_size, y_dim)
        u = self.base_distribution.sample((num_samples, x.size(0)))
        # 2) flatten for batch processing
        batch = x.size(0)
        u_rep = u.reshape(-1, u.size(-1))            # (num_samples*batch, y_dim)
        # 3) repeat context
        x_rep = x.unsqueeze(0).expand(num_samples, batch, x.size(-1))
        x_rep = x_rep.reshape(-1, x.size(-1))        # (num_samples*batch, x_dim)
        # 4) generate
        y_rep = self.transformation(x_rep, u_rep)    # (num_samples*batch, y_dim)
        # 5) reshape back
        return y_rep.view(num_samples, batch, -1)

    def cocycle(self,
               x1: torch.Tensor,
               x2: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
        u = self.inverse_transformation(x2, y)
        return self.transformation(x1, u)

    def cocycle_outer(self,
                      x1: torch.Tensor,
                      x2: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor of shape (M, N, y_dim) where
           v[i,j] = cocycle(x1[i], x2[j], y[j])
        """
        M, N = x1.size(0), y.size(0)
        # 1) abduct each y[j] -> u[j]
        u = self.inverse_transformation(x2, y)       # (N, y_dim)
        # 2) build the (i,j) grid
        x1_rep = x1.unsqueeze(1).expand(M, N, x1.size(1)).reshape(M * N, -1)
        u_rep  = u.unsqueeze(0).expand(M, N, u.size(1)).reshape(M * N, -1)
        # 3) generate all pairs
        v_rep  = self.transformation(x1_rep, u_rep)  # (M*N, y_dim)
        # 4) reshape back
        return v_rep.view(M, N, -1)

