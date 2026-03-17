import copy
import torch
from typing import Optional

from causal_cocycle.rkhs_estimation import FunctionalRegressor


class RKHSWeightEstimator:
    """
    High-level wrapper for learning linear smoother weights on covariates
    ``(X, Z)`` and returning matrices

        W[i, j] = beta_j(X_query[i], Z_query[i]).

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
        self._Z: Optional[torch.Tensor] = None
        self._Y: Optional[torch.Tensor] = None
        self._U: Optional[torch.Tensor] = None

    @staticmethod
    def _as_2d(name: str, value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 1:
            return value.unsqueeze(-1)
        if value.dim() != 2:
            raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(value.shape)}.")
        return value

    def _stack_inputs(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        X = self._as_2d("X", X)
        Z = self._as_2d("Z", Z)
        if X.shape[0] != Z.shape[0]:
            raise ValueError(
                f"X and Z must have the same batch size, got {X.shape[0]} and {Z.shape[0]}."
            )
        return torch.cat((X, Z), dim=-1)

    def tune(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
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

            ||psi(Y) - m_hat(X, Z)||_H^2.
        """
        if kernel_y is not None:
            self.kernel_y = kernel_y

        if self.kernel_y is None:
            raise ValueError("kernel_y must be provided either at init or in tune().")

        self.regressor = FunctionalRegressor(self.functional, kernel_y=self.kernel_y)
        U = self._stack_inputs(X, Z)
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

    def fit(self, X: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor):
        """
        Store the support set and fit the smoother on ``U = [X, Z]``.
        """
        X = self._as_2d("X", X)
        Z = self._as_2d("Z", Z)
        Y = self._as_2d("Y", Y)
        U = self._stack_inputs(X, Z)

        self._X = X
        self._Z = Z
        self._Y = Y
        self._U = U
        self.functional.fit(Y, U)
        return self

    def fit_tune(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
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
            Z,
            Y,
            kernel_y=kernel_y,
            maxiter=maxiter,
            nfold=nfold,
            learn_rate=learn_rate,
            subsamples=subsamples,
            print_=print_,
            generator=generator,
        )
        return self.fit(X, Z, Y)

    def weights(self, X_query: torch.Tensor, Z_query: torch.Tensor) -> torch.Tensor:
        """
        Return the smoother coefficient matrix of shape

            (n_query, n_support).
        """
        if self._U is None:
            raise ValueError("Estimator must be fit before calling weights().")

        U_query = self._stack_inputs(X_query, Z_query)
        return self.functional.linear_weights(U_query)

    def training_weights(self) -> torch.Tensor:
        """
        Return in-sample smoother weights of shape ``(n_support, n_support)``.
        """
        if self._U is None:
            raise ValueError("Estimator must be fit before calling training_weights().")
        return self.functional.training_weights()

    def batch_weight_fn(self):
        """
        Return a callable suitable for ``WCMMD_V`` in the current minibatch regime.

        The callable expects either:
        - a tensor ``U_batch`` containing concatenated covariates, or
        - a dict with entries ``'X'`` and ``'Z'``.

        It fits a fresh copy of the smoother on the batch and returns the
        batch-local weight matrix of shape ``(n_batch, n_batch)``.
        """

        def _weight_fn(inputs):
            if isinstance(inputs, dict):
                if "X" not in inputs or "Z" not in inputs:
                    raise ValueError("Batch dict inputs must contain keys 'X' and 'Z'.")
                X_batch = inputs["X"]
                Z_batch = inputs["Z"]
            else:
                raise ValueError(
                    "batch_weight_fn currently expects dict inputs with keys 'X' and 'Z'."
                )

            batch_functional = copy.deepcopy(self.functional)
            U_batch = self._stack_inputs(X_batch, Z_batch)

            # Any Y-like argument with the correct first dimension works here because
            # linear_weights depend only on the support covariates and tuned
            # hyperparameters. Using zeros avoids imposing an outcome dependency.
            dummy_y = torch.zeros(
                (U_batch.shape[0], 1),
                device=U_batch.device,
                dtype=U_batch.dtype,
            )
            batch_functional.fit(dummy_y, U_batch)
            return batch_functional.training_weights()

        return _weight_fn
