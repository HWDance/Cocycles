# -*- coding: utf-8 -*-
"""
Optimisation module for coboundary models / CTMs / BCMs with flows,
updated to use a standard PyTorch training loop with DataLoader,
an optional validation DataLoader, and the ability to optimize additional
parameters (e.g., for a parameterized base distribution) with a custom LR,
and supporting separate training and validation loss functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import inspect
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# For handling dictionaries
def index_select_inputs(inputs, indices):
    if isinstance(inputs, dict):
        return {k: v[indices] for k, v in inputs.items()}
    else:
        return inputs[indices]
def move_to_device(inputs, device):
    if isinstance(inputs, dict):
        return {k: v.to(device) for k, v in inputs.items()}
    else:
        return inputs.to(device)

def optimise(model,
                    loss_tr,
                    inputs,
                    outputs, 
                    inputs_val=None,
                    outputs_val=None,
                    learn_rate=1e-3, 
                    epochs=100,
                    weight_decay=0,
                    batch_size=1024,
                    val_batch_size=1024,
                    scheduler=False, 
                    schedule_milestone=10, 
                    lr_mult=0.90, 
                    print_=True, 
                    plot=False,
                    likelihood_param_opt=False,
                    likelihood_param_lr=0.01,
                    loss_val=None):
    """    
    Parameters:
    -----------
    model : nn.Module
        The model to be trained.
    loss_tr : callable or nn.Module
        The loss function used for training. Expected to be called as: loss = loss_tr(model, x, y).
    inputs : torch.Tensor
        Training inputs (N x D).
    outputs : torch.Tensor
        Training outputs (N x P).
    inputs_val : torch.Tensor, optional
        Validation inputs.
    outputs_val : torch.Tensor, optional
        Validation outputs.
    learn_rate : float
        Learning rate for model parameters.
    epochs : int
        Number of epochs to train.
    weight_decay : float
        Weight decay (L2 regularization) factor.
    batch_size : int
        Batch size for training.
    val_batch_size : int
        Batch size for validation.
    scheduler : bool
        Whether to use a StepLR scheduler.
    schedule_milestone : int
        Epoch interval at which to decay the learning rate.
    lr_mult : float
        Multiplicative factor for learning rate decay.
    print_ : bool
        Whether to print training progress.
    plot : bool
        Whether to plot the loss curve during training.
    likelihood_param_opt : bool
        If True, also optimize parameters of loss_tr (e.g., a parameterized base distribution).
    likelihood_param_lr : float
        Learning rate to use for the additional parameters.
    loss_val : callable or nn.Module, optional
        The loss function used for validation. If None, loss_tr is used.
    
    Returns:
    --------
    losses : list of float
        List of average training losses per epoch.
    avg_val_loss : float or None
        Average validation loss if validation data is provided; otherwise, None.
    """
    
    # Use loss_tr for validation if loss_val not provided.
    if loss_val is None:
        loss_val = loss_tr

    device = next(model.parameters()).device
    model.train()

    # Move dataset to device.
    inputs = move_to_device(inputs, device)
    outputs = move_to_device(outputs, device)
    if inputs_val is not None and outputs_val is not None:
        inputs_val = move_to_device(inputs_val, device)
        outputs_val = move_to_device(outputs_val, device)

    N = outputs.size(0)

    # Build parameter groups
    param_list = list(model.parameters())
    param_groups = [{'params': param_list, 'lr': learn_rate}]
    if likelihood_param_opt:
        loss_param_list = list(loss_tr.parameters())
        param_groups.append({'params': loss_param_list, 'lr': likelihood_param_lr})

    optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=schedule_milestone, gamma=lr_mult)
    else:
        lr_scheduler = None

    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        # For each epoch, decide on a fixed number of iterations.
        # Here we take as many iterations as fit the dataset (you can adjust this).
        num_batches = max(N // batch_size , 1)
        for i in range(num_batches):
            # Sample a new batch of indices (without replacement for the current batch).
            indices = torch.randperm(N)[:batch_size]
            x_batch = index_select_inputs(inputs, indices)
            y_batch = outputs[indices]

            optimizer.zero_grad()
            loss_value = loss_tr(model, x_batch, y_batch)
            loss_value.backward()
            optimizer.step()
            total_loss += loss_value.item() * y_batch.size(0)

        epoch_loss = total_loss / (num_batches * batch_size)
        losses.append(epoch_loss)

        if scheduler:
            lr_scheduler.step()

        if print_:
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}")

        if plot and (epoch+1) % 10 == 0:
            plt.figure()
            plt.plot(range(1, epoch+2), losses, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("Loss Curve")
            plt.show()

    avg_val_loss = None
    if inputs_val is not None and outputs_val is not None:
        model.eval()
        with torch.no_grad():
            if outputs_val.size(0) <= val_batch_size:
                # If the entire validation set fits in one batch, compute loss on all of it.
                v_loss = loss_val(model, inputs_val, outputs_val)
                avg_val_loss = v_loss.item()
            else:
                total_val_loss = 0.0
                num_val_batches = max(outputs_val.size(0) // val_batch_size, 1)
                for i in range(num_val_batches):
                    indices = torch.randperm(outputs_val.size(0))[:val_batch_size]
                    x_val_batch = index_select_inputs(inputs_val,indices)
                    y_val_batch = outputs_val[indices]
                    v_loss = loss_val(model, x_val_batch, y_val_batch)
                    total_val_loss += v_loss.item() * y_val_batch.size(0)
                avg_val_loss = total_val_loss / (num_val_batches * val_batch_size)
        if print_:
            print(f"Validation Loss: {avg_val_loss:.4f}")


    return losses, avg_val_loss

def get_CV_splits(X, folds=5):
    """
    Splits input X into 'folds' parts for cross-validation.
    X can be a torch.Tensor or a dict of tensors (with matching first dimensions).
    
    Returns a list of (X_train, X_val) pairs, preserving the input structure.
    """
    if isinstance(X, dict):
        n = list(X.values())[0].size(0)  # assume all values have same batch dim
    else:
        n = X.size(0)

    n_per_fold = int(np.ceil(n / folds))
    indices = torch.arange(n)
    splits = []

    for i in range(folds):
        idx_val = (indices >= i * n_per_fold) & (indices < (i + 1) * n_per_fold)
        idx_train = ~idx_val
        idx_val = idx_val.nonzero(as_tuple=True)[0]
        idx_train = idx_train.nonzero(as_tuple=True)[0]

        if isinstance(X, dict):
            X_train, X_val = split_inputs(X, idx_train, idx_val)
        else:
            X_train, X_val = X[idx_train], X[idx_val]

        splits.append((X_train, X_val))

    return splits


def get_optimiser_defaults(opt_func):
    argspec = inspect.getfullargspec(opt_func)
    defaults = argspec.defaults if argspec.defaults is not None else ()
    default_names = argspec.args[-len(defaults):] if defaults else []
    return dict(zip(default_names, defaults))

def split_inputs(inputs, idx_train, idx_val):
    if isinstance(inputs, dict):
        return (
            {k: v[idx_train] for k, v in inputs.items()},
            {k: v[idx_val] for k, v in inputs.items()}
        )
    else:
        return inputs[idx_train], inputs[idx_val]

def validate(models, loss, inputs, outputs, loss_val=None, method="fixed", train_val_split=0.8,
             opt_kwargs=None, hyper_kwargs=None, choose_best_model="overall", retrain=True):
    """
    Validate a list of models using either k-fold cross-validation or a fixed train/validation split.
    
    Parameters:
    -----------
    models : list of nn.Module
        The models to be validated.
    loss : callable or nn.Module
        The loss function used for training.
    inputs : torch.Tensor
        Input data of shape (N x D).
    outputs : torch.Tensor
        Output data of shape (N x P).
    loss_val : callable or nn.Module, optional
        A loss function for validation. If None, 'loss' is used.
    method : str, default "fixed"
        Either "CV" for k-fold cross-validation or "fixed" for a fixed train/val split.
    train_val_split : float, default 0.8
        Fraction of data to use for training.
    opt_kwargs : dict, optional
        A dictionary of additional keyword arguments to pass to the optimise() function.
        Any parameters not specified here will take the defaults defined in optimise().
    hyper_kwargs : dict or list of dict, optional
        Additional hyperparameter settings for optimiser arguments.
        If a single dict is provided, it is applied to all models; if a list is provided,
        it should have the same length as 'models'.
    choose_best_model : str, default "overall"
        "overall" to choose the model with the lowest average validation loss across folds,
        or "per fold" to select the best model on each fold.
    retrain : bool, default True
        If True and choose_best_model=="overall", retrain the best model on the full dataset.
    
    Returns:
    --------
    final_model : nn.Module or list of nn.Module
        The best model(s) selected.
    best_loss : float or list of float
        The corresponding validation loss (or a list of losses if choose_best_model=="per fold").
    """

    # Extract default optimiser options from the optimise function.
    default_opts = get_optimiser_defaults(optimise)
    if opt_kwargs is None:
        opt_kwargs = {}
    final_opts = default_opts.copy()
    final_opts.update(opt_kwargs)

    # Use loss for validation if not provided.
    if loss_val is None:
        loss_val = loss
    final_opts['loss_val'] = loss_val
    
    # Prepare data splits.
    n = outputs.size(0)
    if method == "CV":
        folds = int(1/(1-train_val_split))
        input_splits = get_CV_splits(inputs,folds)
        output_splits = get_CV_splits(outputs,folds)
    else:
        folds = 1
        n_train = int(train_val_split * n)
        idx_train = torch.arange(n_train)
        idx_val = torch.arange(n_train, n)
        in_train, in_val = split_inputs(inputs, idx_train, idx_val)
        out_train, out_val = outputs[idx_train], outputs[idx_val]
        input_splits = [[in_train, in_val]]
        output_splits = [[out_train, out_val]]

    
    # Ensure hyper_kwargs is a list of dictionaries (one per model).
    if hyper_kwargs is None:
        hyper_kwargs = [{}] * len(models)
    elif isinstance(hyper_kwargs, dict):
        hyper_kwargs = [hyper_kwargs] * len(models)
    else:
        assert len(hyper_kwargs) == len(models), "Length of hyper_kwargs must match number of models."

    # Matrix to store validation losses: shape (num_models, folds).
    val_losses = torch.zeros((len(models), folds))
    models_store = []  # For each model, store list of fold-trained candidates.
    
    for m, base_model in enumerate(models):
        fold_models = []
        for k in range(folds):
            # Deep copy the base model.
            model_copy = copy.deepcopy(base_model)
            # Get train/validation splits for fold k.
            inputs_train, inputs_val = input_splits[k]
            outputs_train, outputs_val = output_splits[k]
            
            # Merge the general optimizer options with hyperparameter overrides for this model.
            current_opts = final_opts.copy()
            current_opts.update(hyper_kwargs[m])

            # Update with the fold-specific validation splits:
            current_opts['inputs_val'] = inputs_val
            current_opts['outputs_val'] = outputs_val
            
            # Call the optimise function.
            # Note: We assume optimise has the following signature:
            #   optimise(model, loss_tr, inputs, outputs, **current_opts)
            train_losses, curr_val_loss = optimise(model_copy, loss, inputs_train, outputs_train,
                                                   **current_opts)
            val_losses[m, k] = curr_val_loss if curr_val_loss is not None else float('inf')
            fold_models.append(model_copy)
            print(f"Model {m}, Fold {k}: Validation Loss = {curr_val_loss:.4f}")
        models_store.append(fold_models)
    
    # Choose best model(s).
    val_losses[torch.isnan(val_losses)] = float('inf')
    if torch.isinf(val_losses).all():
        raise RuntimeError("All models produced NaN losses.")
    if choose_best_model == "overall":
        mean_val_losses = val_losses.mean(dim=1)
        best_index = torch.argmin(mean_val_losses).item()
        best_model_candidates = models_store[best_index]
        best_avg_loss = mean_val_losses[best_index].item()
        print(f"Best overall model index: {best_index} with average validation loss {best_avg_loss:.4f}")
        final_model = best_model_candidates  # List of candidate models from each fold.
        # Optionally retrain on the full dataset.
        if retrain:
            print("Retraining best model on full dataset...")
            retrain_opts = final_opts.copy()
            retrain_opts.update(hyper_kwargs[best_index])
            retrain_opts['inputs_val'] = None
            retrain_opts['outputs_val'] = None
            # Deep copy the candidate and retrain using the full dataset.
            final_model = copy.deepcopy(models[best_index])
            _, _ = optimise(final_model, loss, inputs, outputs, **retrain_opts)
    elif choose_best_model == "per fold":
        final_model = []
        best_losses = []
        best_index = []
        
        for k in range(folds):
            fold_losses = val_losses[:, k]
            best_idx = torch.argmin(fold_losses).item()
            final_model.append(models_store[best_idx][k])
            best_index.append(best_idx)
            best_losses.append(fold_losses[best_idx].item())
        best_avg_loss = best_losses
        print(f"Best model per fold selected with losses: {best_losses}")
    else:
        raise ValueError("choose_best_model must be either 'overall' or 'per fold'")
    
    return final_model, (best_index, best_avg_loss)

    
                

    