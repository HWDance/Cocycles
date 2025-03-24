# conditional_expectation_regressor.py

import torch
from torch.utils.data import Dataset, DataLoader

class RegressionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ConditionalExpectationRegressor:
    def __init__(self, functional):
        self.functional = functional
   
    def get_CV_splits(self, X, Y, nfolds):
        """
        Returns list of (train_indices, val_indices) for cross-validation.
        Compatible with random mini-batches.
        """
        n = len(X)
        nfolds = min(n, nfolds)  # Ensure we don’t over-split
    
        indices = torch.randperm(n)
        fold_sizes = [(n + i) // nfolds for i in range(nfolds)]  # near-equal partitions
    
        folds = []
        start = 0
        for size in fold_sizes:
            end = start + size
            folds.append(indices[start:end])
            start = end
    
        splits = []
        for i in range(nfolds):
            val_idx = folds[i]
            train_idx = torch.cat([folds[j] for j in range(nfolds) if j != i])
            splits.append((train_idx, val_idx))
        return splits


    def evaluate_CV_loss_fixed(self, X, Y, splits, norm=2):
        total_loss = 0.0
        for train_idx, val_idx in splits:
            Xtrain, Ytrain = X[train_idx], Y[train_idx]
            Xval, Yval = X[val_idx], Y[val_idx]
            Ypred = self.functional(Ytrain, Xtrain, Xval)
            total_loss += torch.mean((Yval - Ypred) ** norm)
        return total_loss / len(splits)


    def get_subsample(self, X, Y, subsamples):
        idx = torch.randperm(len(X))[:subsamples]
        return X[idx], Y[idx]
    
    def optimise(self, X, Y, maxiter=100, nfold=5, 
                 learn_rate=1e-2, batch_size=None, 
                 norm=2, print_=False):
        """
        Optimizes the kernel hyperparameters over a fixed number of iterations.
        Cross-validation splits are fixed throughout.
        """
        X = X.float()
        Y = Y.float()
        device = next(self.functional.parameters()).device
        X, Y = X.to(device), Y.to(device)
    
        # Fix CV splits once
        fixed_splits = self.get_CV_splits(X, Y, nfolds=nfold)
    
        dataset = RegressionDataset(X, Y)
        loader = DataLoader(dataset, batch_size=batch_size or len(dataset), shuffle=True)
    
        optimizer = torch.optim.Adam(self.functional.parameters(), lr=learn_rate)
        losses = []
    
        for i in range(maxiter):
            epoch_loss = 0.0
    
            for X_batch, Y_batch in loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
    
                loss = self.evaluate_CV_loss_fixed(X_batch, Y_batch, fixed_splits, norm)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
    
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
    
            if print_ and i % 10 == 0:
                print(f"[iter {i}] avg loss: {avg_loss:.6f}")
    
        return torch.tensor(losses)

    def CVgridsearch(self, X, Y, nfold=5, subsample=False, subsamples=1000, 
                     hyper_grid=[], norm=2):
        X = X.float()
        Y = Y.float()
        device = next(self.functional.parameters()).device
        X = X.to(device)
        Y = Y.to(device)

        if subsample and subsamples < len(Y):
            X, Y = self.get_subsample(X, Y, subsamples)

        splits = self.get_CV_splits(X, Y, nfold)
        losses = torch.zeros(len(hyper_grid))

        for j, hyperparams in enumerate(hyper_grid):
            # Assign hyperparameters manually
            for param, new_val in zip(self.functional.hyperparameters, hyperparams):
                param.data = new_val.data.clone()

            total_loss = 0.0
            for train_idx, val_idx in splits:
                Xtrain, Ytrain = X[train_idx], Y[train_idx]
                Xval, Yval = X[val_idx], Y[val_idx]
                Ypred = self.functional(Ytrain, Xtrain, Xval)
                total_loss += torch.mean((Yval - Ypred) ** norm)

            losses[j] = total_loss / nfold

        best_idx = torch.argmin(losses)
        best_hyperparams = hyper_grid[best_idx]

        # Set best hyperparameters
        for param, best_val in zip(self.functional.hyperparameters, best_hyperparams):
            param.data = best_val.data.clone()

        return losses

def predict(self, Xtrain, Ytrain, Xtest):
    return self.functional(Ytrain, Xtrain, Xtest)    
    

