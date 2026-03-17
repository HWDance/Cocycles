import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def _kernel_device(kernel) -> torch.device:
    if hasattr(kernel, "log_lengthscale") and torch.is_tensor(kernel.log_lengthscale):
        return kernel.log_lengthscale.device
    params = list(kernel.parameters()) if isinstance(kernel, nn.Module) else []
    if params:
        return params[0].device
    return torch.device("cpu")


class Functional(nn.Module):
    """
    Base class for linear smoothers of the form

        m_hat(x) = sum_j W(x)_j psi(Y_j),

    where ``linear_weights(Xtest)`` returns the matrix ``W``.
    """

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.to(_kernel_device(kernel))
        self._Xtrain: Optional[torch.Tensor] = None
        self._Ytrain: Optional[torch.Tensor] = None

    def fit(self, Ytrain: torch.Tensor, Xtrain: torch.Tensor):
        self._Xtrain = Xtrain
        self._Ytrain = Ytrain.view(Ytrain.shape[0], -1)
        self._fit_impl()
        return self

    def _fit_impl(self):
        pass

    def linear_weights(self, Xtest: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, Xtest: torch.Tensor) -> torch.Tensor:
        W = self.linear_weights(Xtest)
        return W @ self._Ytrain

    @torch.no_grad()
    def training_weights(self) -> torch.Tensor:
        return self.linear_weights(self._Xtrain)


class NWFunctional(Functional):
    """
    Row-normalized kernel smoother:

        W_ij = K(Xtest_i, Xtrain_j) / sum_j K(Xtest_i, Xtrain_j).
    """

    def __init__(self, kernel, eps: float = 1e-12):
        super().__init__(kernel)
        self.eps = eps

    @property
    def hyperparameters(self):
        return [self.kernel.log_lengthscale] if hasattr(self.kernel, "log_lengthscale") else []

    def _fit_impl(self):
        pass

    def linear_weights(self, Xtest: torch.Tensor) -> torch.Tensor:
        K = self.kernel.get_gram(Xtest, self._Xtrain)
        denom = K.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return K / denom


class KRRFunctional(Functional):
    """
    Kernel ridge regression smoother:

        W(x, X) = K(x, X) (K(X, X) + lambda I)^(-1),

    computed without forming explicit inverses.
    """

    def __init__(self, kernel, penalty: float = 1e-2, jitter: float = 1e-6):
        super().__init__(kernel)
        self.log_penalty = nn.Parameter(torch.tensor(np.log(penalty), dtype=torch.float32))
        self.jitter = jitter
        self._L: Optional[torch.Tensor] = None
        self.alpha_: Optional[torch.Tensor] = None

    @property
    def penalty(self):
        return torch.exp(self.log_penalty)

    @property
    def hyperparameters(self):
        params = [self.log_penalty]
        if hasattr(self.kernel, "log_lengthscale"):
            params.append(self.kernel.log_lengthscale)
        return params

    def _fit_impl(self):
        X = self._Xtrain
        K_xx = self.kernel.get_gram(X, X)
        n = K_xx.shape[0]
        eye = torch.eye(n, device=K_xx.device, dtype=K_xx.dtype)
        K_reg = K_xx + (self.penalty + self.jitter) * eye
        self._L = torch.linalg.cholesky(K_reg)
        self.alpha_ = torch.cholesky_solve(self._Ytrain, self._L)

    def linear_weights(self, Xtest: torch.Tensor) -> torch.Tensor:
        K_xX = self.kernel.get_gram(Xtest, self._Xtrain)
        Zt = torch.cholesky_solve(K_xX.transpose(0, 1), self._L)
        return Zt.transpose(0, 1)

    @torch.no_grad()
    def predict(self, Xtest: torch.Tensor) -> torch.Tensor:
        K_xX = self.kernel.get_gram(Xtest, self._Xtrain)
        return K_xX @ self.alpha_
