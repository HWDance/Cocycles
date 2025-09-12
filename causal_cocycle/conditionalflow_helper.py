import copy
from typing import Callable, List
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from causalflows.flows import CausalFlow


def select_and_train_flow(
    flows: List[CausalFlow],
    X: torch.Tensor,  # context variables
    Y: torch.Tensor,  # feature/target variables
    train_fraction: float = 1.0,
    k_folds: int = 2,
    num_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = None,
    flow_lrs: List[float] = None,
):
    """
    Select and train the best causal flow model, using X as context and Y as features.

    Returns:
        best_flow: trained model
        test_nll: negative log-likelihood on test set (or None)
        best_idx: index of chosen flow
        cv_scores: cross-validation scores for each flow
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, d_y = Y.shape

    # Train/test split
    perm = torch.randperm(N)
    n_train = int(train_fraction * N)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # NLL evaluation
    def eval_nll(model: CausalFlow, X_eval: torch.Tensor, Y_eval: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            X_eval = X_eval.to(device)
            Y_eval = Y_eval.to(device)
            dist = model(X_eval)
            return -dist.log_prob(Y_eval).mean().item()

    # Cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True)
    cv_scores: List[float] = []

    for i, flow in enumerate(flows):
        fold_scores: List[float] = []
        lr_used = flow_lrs[i] if flow_lrs is not None else lr

        for train_f, val_f in kf.split(Y_train):
            model = copy.deepcopy(flow).to(device).train()
            dataset = TensorDataset(X_train[train_f], Y_train[train_f])
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optim = torch.optim.Adam(model.parameters(), lr=lr_used)

            # Train
            for _ in range(num_epochs):
                for x_b, y_b in loader:
                    x_b = x_b.to(device)
                    y_b = y_b.to(device)
                    dist = model(x_b)
                    loss = -dist.log_prob(y_b).mean()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

            # Validate
            fold_scores.append(eval_nll(model,
                                        X_train[val_f],
                                        Y_train[val_f]))
        cv_scores.append(sum(fold_scores)/len(fold_scores))

    # Select best flow
    best_idx = int(torch.argmin(torch.tensor(cv_scores)))
    best_flow = flows[best_idx]

    # Retrain best flow on all data
    best_flow = copy.deepcopy(best_flow).to(device).train()
    dataset_all = TensorDataset(X, Y)
    loader_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(best_flow.parameters(), lr=lr)
    for _ in range(num_epochs):
        for x_b, y_b in loader_all:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            dist = best_flow(x_b)
            loss = -dist.log_prob(y_b).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

    # Test NLL
    test_nll = None
    if train_fraction < 1.0:
        test_nll = eval_nll(best_flow, X_test, Y_test)

    return best_flow, test_nll, best_idx, cv_scores


def sample_do(
    flow: CausalFlow,
    X: torch.Tensor,
    index: int,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_shape: torch.Size = torch.Size([1]),
) -> torch.Tensor:
    """
    Interventional sampling under context X (do on context).
    """
    # Number of context samples to draw
    num_samples = sample_shape[0]
    N = X.shape[0]
    # Randomly select context indices with replacement
    idx = torch.randint(low=0, high=N, size=(num_samples,), device=X.device)
    X_sub = X[idx]
    # Apply intervention to selected context rows
    X_do = copy.deepcopy(X_sub)
    X_do[...,index] = intervention_fn(X_do[...,index])
    # Build distribution under intervened context
    dist = flow(X_do)
    # Sample latent noise and map to output (maintaining sample_shape)
    z = dist.base.sample().unsqueeze(-1)
    y = dist.transform.inv(z)
    return y


def sample_cf(
    flow: CausalFlow,
    X: torch.Tensor,
    Y: torch.Tensor,
    index: int,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Counterfactual sampling under context X for observed Y_obs.

    Steps:
      1. Intervene on context to get X_cf
      2. Compute latent U from observed Y under original context
      3. Map U through new context's invert transform to get Y_cf
    """
    # Intervene on context
    X_cf = copy.deepcopy(X)
    X_cf[...,index] = intervention_fn(X_cf[...,index])    # Original and counterfactual distributions
    dist_obs = flow(X)
    dist_cf = flow(X_cf)
    # Obtain latent representation of observed outcomes
    U = dist_obs.transform(Y)
    # Generate counterfactual outcomes
    Y_cf = dist_cf.transform.inv(U)
    return Y_cf
