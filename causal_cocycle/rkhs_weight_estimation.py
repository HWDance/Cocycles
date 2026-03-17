import torch
from typing import Optional

from causal_cocycle.rkhs_estimation import FunctionalRegressor


class RKHSWeightEstimator:
    """
    High-level wrapper for learning fixed linear smoother weights on CME
    conditioning covariates ``(X, C)`` and returning the full in-sample
    weight matrix

        W[i, j] = beta_j(X_query[i], C_query[i]).

    The underlying smoother is any functional exposing

        fit(Ytrain, Xtrain)
        linear_weights(Xtest)

    such as ``NWFunctional`` or ``KRRFunctional`` from
    ``causal_cocycle.rkhs_functionals``.
    """

    def __init__(self, functional, kernel_y=None):
        self.functional = functional
        self.kernel_y = kernel_y
        self.regressor = (
            FunctionalRegressor(functional, kernel_y=kernel_y)
            if kernel_y is not None
            else None
        )
        self._X: Optional[torch.Tensor] = None
        self._C: Optional[torch.Tensor] = None
        self._Y: Optional[torch.Tensor] = None
        self._U: Optional[torch.Tensor] = None

    @staticmethod
    def _as_2d(name: str, value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 1:
            return value.unsqueeze(-1)
        if value.dim() != 2:
            raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(value.shape)}.")
        return value

    def _stack_inputs(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        X = self._as_2d("X", X)
        C = self._as_2d("C", C)
        if X.shape[0] != C.shape[0]:
            raise ValueError(
                f"X and C must have the same batch size, got {X.shape[0]} and {C.shape[0]}."
            )
        return torch.cat((X, C), dim=-1)

    def tune(
        self,
        X: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        kernel_y=None,
        maxiter: int = 100,
        nfold: int = 5,
        learn_rate: float = 1e-2,
        subsamples: Optional[int] = None,
        print_: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Tune smoother hyperparameters by RKHS cross-validation loss

            ||psi(Y) - m_hat(X, C)||_H^2.
        """
        if kernel_y is not None:
            self.kernel_y = kernel_y

        if self.kernel_y is None:
            raise ValueError("kernel_y must be provided either at init or in tune().")

        self.regressor = FunctionalRegressor(self.functional, kernel_y=self.kernel_y)
        U = self._stack_inputs(X, C)
        losses = self.regressor.optimise(
            U,
            self._as_2d("Y", Y),
            maxiter=maxiter,
            nfold=nfold,
            learn_rate=learn_rate,
            subsamples=subsamples,
            print_=print_,
            generator=generator,
        )
        return losses

    def fit(self, X: torch.Tensor, C: torch.Tensor, Y: torch.Tensor):
        """
        Store the support set and fit the smoother on ``U = [X, C]``.

        This is the intended preprocessing step for ``WCMMD_V``:
        fit once on the full support set, then use ``training_weights()`` to
        obtain the full fixed weight matrix ``W`` before cocycle training.
        """
        X = self._as_2d("X", X)
        C = self._as_2d("C", C)
        Y = self._as_2d("Y", Y)
        U = self._stack_inputs(X, C)

        self._X = X
        self._C = C
        self._Y = Y
        self._U = U
        self.functional.fit(Y, U)
        return self

    def fit_tune(
        self,
        X: torch.Tensor,
        C: torch.Tensor,
        Y: torch.Tensor,
        kernel_y=None,
        maxiter: int = 100,
        nfold: int = 5,
        learn_rate: float = 1e-2,
        subsamples: Optional[int] = None,
        print_: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Tune hyperparameters by RKHS CV, then fit on the full support set.
        """
        self.tune(
            X,
            C,
            Y,
            kernel_y=kernel_y,
            maxiter=maxiter,
            nfold=nfold,
            learn_rate=learn_rate,
            subsamples=subsamples,
            print_=print_,
            generator=generator,
        )
        return self.fit(X, C, Y)

    def weights(self, X_query: torch.Tensor, C_query: torch.Tensor) -> torch.Tensor:
        """
        Return the smoother coefficient matrix of shape

            (n_query, n_support).
        """
        if self._U is None:
            raise ValueError("Estimator must be fit before calling weights().")

        U_query = self._stack_inputs(X_query, C_query)
        return self.functional.linear_weights(U_query)

    def training_weights(self) -> torch.Tensor:
        """
        Return the full in-sample fixed weight matrix ``W`` of shape
        ``(n_support, n_support)``. This is the object to pass into
        ``build_loss(..., loss_type="WCMMD_V", weights=W)`` before cocycle
        training.
        """
        if self._U is None:
            raise ValueError("Estimator must be fit before calling training_weights().")
        return self.functional.training_weights()
