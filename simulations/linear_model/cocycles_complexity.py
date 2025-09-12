#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.distributions import Normal,Laplace,Uniform,Gamma, Beta, Cauchy, Distribution
import os
import time
import torch
from copy import deepcopy

# Causal_cocycle imports
from causal_cocycle.model_factory import CocycleFactory
from causal_cocycle.model_factory import FlowFactory
from causal_cocycle.loss_factory import CocycleLossFactory
from causal_cocycle.loss import FlowLoss
from causal_cocycle.optimise_new import validate, optimise
from causal_cocycle.kernels import gaussian_kernel
from lm_config import opt_config, model_config
from causal_cocycle.helper_functions import ks_statistic, wasserstein1_repeat, rmse

def evaluate_models(
    models_dict: dict,
    index_dict: dict,
    X: torch.Tensor,
    Y: torch.Tensor,
    noisedist: Distribution,
    noisetransform: callable,
    sig_noise_ratio: float,
    seed: int = None
) -> dict:
    """
    models_dict: { name: (model, model_type) }
      model_type in {'cocycle','hsic','cmmd_v','cmmd_u','l2','l1','urr'}

    Returns:
      {
        name: {
          'KS_int':   KS between Y_model(X+1) and Y_true(X+1),
          'KS_cf':    KS between (Y_cf-Y) and (Y_true-Y),
          'CF_RMSE':  RMSE between (Y_cf-Y) and (Y_true-Y)
        },
        ...
      }
    """
    N, D = X.shape
    assert(D==1)
    m = 10**5
    _, P = Y.shape
    device = X.device

    # “true” counterfactual data
    X_cf      = X + 1.0  # counterfactual X
    ΔY_true = torch.ones((N,P)) # counterfactual Y

    # “true” interventional data
    Y_true  = (Normal(1,1).sample((m,D)) + 1.0
     + noisetransform(noisedist.sample((m,1))))

    results = {}
    for name, (model, mtype) in models_dict.items():
        # ---- interventional estimate ----
        if mtype in ('hsic','cmmdv','cmmdu'):
            Y_int = model.cocycle(X_cf, X, Y)             # (N,P)
        else:  # l2, l1, urr
            Uhat = model.base_distribution.sample((N,))
            Xhat = Normal(1,1).sample((N,D)) + 1.0
            out = model.transformation(Xhat, Uhat)           # either y or (y,logdet)
            Y_int = out[0] if isinstance(out, tuple) else out

        KS_int = ks_statistic(Y_int[:,0],   Y_true[:,0])

        # ---- counterfactual via cocycle ----
        model.transformer.logdet = False
        Y_cf    = model.cocycle(X_cf, X, Y)     # (N,P)
        ΔY_model= Y_cf - Y                    # (N,P)

        RMSE_cf = rmse(    ΔY_model[:,0],   ΔY_true[:,0])

        results[name] = {
            'KS_int':  KS_int,
            'CF_RMSE': RMSE_cf,
            'index': index_dict[name][0]
        }

    return results


def run_experiment(seed=0, N=1000, noise_dist = "normal"):

    """
    Configs
    """
    # Experimental set up
    D,P = 1,1
    sig_noise_ratio = 1
    
    """
    Main
    """
    
    # Drawing data
    torch.manual_seed(seed)
    X = Normal(1,1).sample((N,D))
    X *= 1/(D)**0.5
    B = torch.ones((D,1))*(torch.linspace(0,D-1,D)<P)[:,None]
    F = X @ B
    if noise_dist == "normal":
        noisedist = Normal(0,1)
        noisetransform = lambda x : x
    elif noise_dist == "rademacher": 
        noisedist = Uniform(-1,1)
        noisetransform = lambda x : torch.sign(x)
    elif noise_dist == "cauchy":
        noisedist = Cauchy(0,1)
        noisetransform = lambda x : x
    elif noise_dist == "gamma":
        noisedist = Gamma(1,1)
        noisetransform = lambda x : x
    elif noise_dist == "inversegamma":
        noisedist = Gamma(1,1)
        noisetransform = lambda x : 1/x
    U = noisetransform(noisedist.sample((N,1)))/sig_noise_ratio**0.5
    Y = F + U

    # Cocycle model construction
    factory = CocycleFactory(1, model_config)
    models, hyper_args = factory.build_models()
    print(f"Constructed {len(models)} candidate cocycle models.")
    
    gauss_config,laplace_config = model_config.copy(),model_config.copy()
    gauss_config['base_distribution_configs'], laplace_config['base_distribution_configs'] = ["Normal"]*4, ["Laplace"]*4
    models_gauss, hyper_args  = FlowFactory(1, gauss_config).build_models()
    models_laplace, hyper_args  = FlowFactory(1, laplace_config).build_models()
    
    models_urr_gauss,models_urr_laplace = deepcopy(models_gauss),deepcopy(models_laplace)
    for i in range(len(models_urr_gauss)):
        models_urr_gauss[i].transformer.logdet = False
        models_urr_laplace[i].transformer.logdet = False

    # Training with L2
    loss= FlowLoss()
    final_model_l2, (best_index_l2, val_loss_l2) = validate(
            models_gauss,
            loss,
            X,
            Y,
            loss_val=loss,
            method="CV",
            train_val_split=0.5,
            opt_kwargs=opt_config,
            hyper_kwargs=hyper_args,
            choose_best_model="overall",
            retrain=True,
        )

    # Training with L1
    loss= FlowLoss()
    final_model_l1, (best_index_l1, val_loss_l1) = validate(
            models_laplace,
            loss,
            X,
            Y,
            loss_val=loss,
            method="CV",
            train_val_split=0.5,
            opt_kwargs=opt_config,
            hyper_kwargs=hyper_args,
            choose_best_model="overall",
            retrain=True,
        )

    # Training with cocycles
    kernel = [gaussian_kernel()] * 2
    loss_factory = CocycleLossFactory(kernel)
    loss= loss_factory.build_loss("CMMD_V", X, Y, subsamples=10**4)
    final_model_cmmdv, (best_index_cmmdv, val_loss_cmmdv) = validate(
            models,
            loss,
            X,
            Y,
            loss_val=loss,
            method="CV",
            train_val_split=0.5,
            opt_kwargs=opt_config,
            hyper_kwargs=hyper_args,
            choose_best_model="overall",
            retrain=True,
        )

    # Training with cocycles
    kernel = [gaussian_kernel()] * 2
    loss_factory = CocycleLossFactory(kernel)
    loss= loss_factory.build_loss("CMMD_U", X, Y, subsamples=10**4)
    final_model_cmmdu, (best_index_cmmdu, val_loss_cmmdu) = validate(
            models,
            loss,
            X,
            Y,
            loss_val=loss,
            method="CV",
            train_val_split=0.5,
            opt_kwargs=opt_config,
            hyper_kwargs=hyper_args,
            choose_best_model="overall",
            retrain=True,
        )

    # Training with hsic
    kernel = [gaussian_kernel()] * 2
    loss_factory = CocycleLossFactory(kernel)
    Uhat = final_model_l2.inverse_transformation(X,Y)[0].detach()
    loss= loss_factory.build_loss("HSIC", X, Uhat, subsamples=10**4)
    final_model_hsic, (best_index_hsic, val_loss_hsic) = validate(
            models,
            loss,
            X,
            Y,
            loss_val=loss,
            method="CV",
            train_val_split=0.5,
            opt_kwargs=opt_config,
            hyper_kwargs=hyper_args,
            choose_best_model="overall",
            retrain=True,
        )    

    # Training with urr
    kernel = [gaussian_kernel()] * 2
    loss_factory = CocycleLossFactory(kernel)
    loss= loss_factory.build_loss("URR", X, Y, subsamples=10**4)
    loss_val= loss_factory.build_loss("URR_N", X, Y, subsamples=10**4)
    final_model_urr, (best_index_urr, val_loss_urr) = validate(
            models_urr_gauss,
            loss,
            X,
            Y,
            loss_val=loss_val,
            method="CV",
            train_val_split=0.5,
            opt_kwargs=opt_config,
            hyper_kwargs=hyper_args,
            choose_best_model="overall",
            retrain=True,
        )

    # Training with urr
    kernel = [gaussian_kernel()] * 2
    loss_factory = CocycleLossFactory(kernel)
    loss= loss_factory.build_loss("URR", X, Y, subsamples=10**4)
    loss_val= loss_factory.build_loss("URR_N", X, Y, subsamples=10**4)
    final_model_urr_l, (best_index_urr_l, val_loss_urr_l) = validate(
            models_urr_laplace,
            loss,
            X,
            Y,
            loss_val=loss_val,
            method="CV",
            train_val_split=0.5,
            opt_kwargs=opt_config,
            hyper_kwargs=hyper_args,
            choose_best_model="overall",
            retrain=True,
        )

    my_models = {
        'L2' : (final_model_l2, 'l2'),
        'L1' : (final_model_l1, 'l1'),
        'URR L2': (final_model_urr,'urr'),
        'URR L1': (final_model_urr_l,'urr'),
        'HSIC': (final_model_hsic,'hsic'),
        'CMMD_V': (final_model_cmmdv,'cmmdv'),
        'CMMD_U': (final_model_cmmdu,'cmmdu'),
    }
    
    my_indexes = {
        'L2' : (best_index_l2, 'l2'),
        'L1' : (best_index_l1, 'l1'),
        'URR L2': (best_index_urr,'urr'),
        'URR L1': (best_index_urr_l,'urr'),
        'HSIC': (best_index_hsic,'hsic'),
        'CMMD_V': (best_index_cmmdv,'cmmdv'),
        'CMMD_U': (best_index_cmmdu,'cmmdu'),
    }
        
    metrics = evaluate_models(
        my_models,
        my_indexes,
        X, Y,
        noisedist, noisetransform,
        sig_noise_ratio=sig_noise_ratio,
        seed=seed
    )

    # Storing noise dist
    metrics['noise_distribution'] = noise_dist
    
    return metrics