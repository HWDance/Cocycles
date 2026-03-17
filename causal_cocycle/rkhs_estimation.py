import torch
from typing import List, Optional, Tuple


class FunctionalRegressor:
    """
    Cross-validation trainer for kernel-based RKHS regression functionals.

    The validation criterion is

        (1 / m) sum_i ||psi(Y_i) - m_hat(X_i)||_H^2,

    computed in closed form using the output kernel ``kernel_y`` and the
    functional's linear smoother weights.
    """

    def __init__(self, functional, kernel_y: Optional[object] = None):
        self.functional = functional
        if kernel_y is not None:
            self.kernel_y = kernel_y
        else:
            ky = getattr(functional, "kernel_y", None)
            if ky is None:
                raise ValueError(
                    "A Y-kernel `kernel_y` is required. Pass it to FunctionalRegressor "
                    "or define `functional.kernel_y`."
                )
            self.kernel_y = ky

    def get_CV_splits(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        nfolds: int,
        generator: Optional[torch.Generator] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        n = X.shape[0]
        nfolds = int(max(1, min(nfolds, n)))

        if generator is None:
            perm = torch.randperm(n, device=X.device)
        else:
            perm = torch.randperm(n, generator=generator, device=X.device)

        fold_sizes = [(n + i) // nfolds for i in range(nfolds)]
        folds = []
        start = 0
        for size in fold_sizes:
            end = start + size
            folds.append(perm[start:end])
            start = end

        splits = []
        for k in range(nfolds):
            val_idx = folds[k]
            train_idx = torch.cat([folds[j] for j in range(nfolds) if j != k], dim=0)
            splits.append((train_idx, val_idx))
        return splits

    def _kernel_val_loss_fold(
        self,
        Xtr: torch.Tensor,
        Ytr: torch.Tensor,
        Xva: torch.Tensor,
        Yva: torch.Tensor,
    ) -> torch.Tensor:
        self.functional.fit(Ytr, Xtr)
        S = self.functional.linear_weights(Xva)

        kernel_y = self.kernel_y
        K_TT = kernel_y.get_gram(Ytr, Ytr)
        K_VT = kernel_y.get_gram(Yva, Ytr)
        K_VV = kernel_y.get_gram(Yva, Yva)

        m_va = Xva.shape[0]
        term_A = torch.diag(K_VV).sum()
        term_B = torch.sum(S * K_VT)
        term_C = torch.trace(S @ K_TT @ S.T)
        return (term_A - 2.0 * term_B + term_C) / m_va

    def evaluate_CV_loss_fixed(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        splits: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        total = 0.0
        for train_idx, val_idx in splits:
            Xtr, Ytr = X[train_idx], Y[train_idx]
            Xva, Yva = X[val_idx], Y[val_idx]
            total = total + self._kernel_val_loss_fold(Xtr, Ytr, Xva, Yva)
        return total / len(splits)

    def subsample_train_fold(
        self,
        Xtr: torch.Tensor,
        Ytr: torch.Tensor,
        subsamples: Optional[int],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ntr = Xtr.shape[0]
        if subsamples is None or subsamples >= ntr:
            return Xtr, Ytr
        if generator is None:
            perm = torch.randperm(ntr, device=Xtr.device)[:subsamples]
        else:
            perm = torch.randperm(ntr, generator=generator, device=Xtr.device)[:subsamples]
        return Xtr[perm], Ytr[perm]

    def optimise(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        maxiter: int = 100,
        nfold: int = 5,
        learn_rate: float = 1e-2,
        subsamples: Optional[int] = None,
        print_: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        X = X.float()
        Y = Y.float()

        device = next(self.functional.parameters()).device
        X, Y = X.to(device), Y.to(device)

        optimizer = torch.optim.Adam(self.functional.parameters(), lr=learn_rate)
        losses: List[float] = []
        splits = self.get_CV_splits(X, Y, nfolds=nfold, generator=generator)

        for it in range(maxiter):
            total = 0.0
            for train_idx, val_idx in splits:
                Xtr, Ytr = X[train_idx], Y[train_idx]
                Xtr, Ytr = self.subsample_train_fold(Xtr, Ytr, subsamples, generator=generator)
                Xva, Yva = X[val_idx], Y[val_idx]
                total = total + self._kernel_val_loss_fold(Xtr, Ytr, Xva, Yva)

            avg_loss = total / len(splits)

            optimizer.zero_grad(set_to_none=True)
            avg_loss.backward()
            optimizer.step()

            losses.append(float(avg_loss.detach().cpu()))
            if print_ and (it % 10 == 0):
                ls = getattr(self.functional.kernel, "lengthscale", None)
                ls_str = f"{ls.detach().cpu().numpy()}" if torch.is_tensor(ls) else "N/A"
                penalty = getattr(self.functional, "log_penalty", None)
                penalty_str = (
                    f" | log_penalty: {float(penalty.detach().cpu())}"
                    if torch.is_tensor(penalty)
                    else ""
                )
                print(
                    f"[iter {it}] avg CV RKHS loss: {losses[-1]:.6f} | lengthscale: {ls_str}{penalty_str}"
                )

        return losses

    def CVgridsearch(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        nfold: int = 5,
        hyper_grid: List[List[torch.Tensor]] = (),
        subsample: bool = False,
        subsamples: int = 1000,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        X = X.float()
        Y = Y.float()

        device = next(self.functional.parameters()).device
        X, Y = X.to(device), Y.to(device)

        if subsample and subsamples < X.shape[0]:
            if generator is None:
                perm = torch.randperm(X.shape[0], device=device)[:subsamples]
            else:
                perm = torch.randperm(X.shape[0], generator=generator, device=device)[:subsamples]
            X, Y = X[perm], Y[perm]

        splits = self.get_CV_splits(X, Y, nfolds=nfold, generator=generator)
        losses = torch.zeros(len(hyper_grid), device=device, dtype=X.dtype)

        for j, hyperparams in enumerate(hyper_grid):
            with torch.no_grad():
                for param, new_val in zip(self.functional.hyperparameters, hyperparams):
                    param.copy_(new_val)

            total = 0.0
            for train_idx, val_idx in splits:
                Xtr, Ytr = X[train_idx], Y[train_idx]
                Xva, Yva = X[val_idx], Y[val_idx]
                total = total + self._kernel_val_loss_fold(Xtr, Ytr, Xva, Yva)

            losses[j] = total / len(splits)

        best_idx = int(torch.argmin(losses).item())
        best_hparams = hyper_grid[best_idx]
        with torch.no_grad():
            for param, best_val in zip(self.functional.hyperparameters, best_hparams):
                param.copy_(best_val)

        print(f"Best CV RKHS loss: {losses[best_idx].item():.6f} at index {best_idx}")
        return losses
