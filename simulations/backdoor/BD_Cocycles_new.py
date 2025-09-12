# Imports
import pandas as pd
import numpy as np
from BD import *

import torch
from torch.distributions import Normal,Gamma,HalfNormal,HalfCauchy
import os
import time
from functools import partial

from causal_cocycle.model import cocycle_model,cocycle_outcome_model
from causal_cocycle.optimise import *
from causal_cocycle.loss_functions import Loss
from causal_cocycle.conditioners import Lin_Conditioner,NN_RELU_Conditioner
from causal_cocycle.transformers import Transformer,Shift_layer,Scale_layer,RQS_layer
from causal_cocycle.helper_functions import likelihood_loss,mmd,propensity_score, empirical_KR
from causal_cocycle.kernels import *


def run_experiment(seed,N,D,flip_prob,splits,cocycle,width,weight_decay):

    name = "{0}_cocycle_width{1}_decay{2}".format(cocycle,width,weight_decay)

    # Treatment effect estimation
    splits = splits

    # DGP set-up
    N = N
    D = D
    Nint = 10**4
    Zcorr = 0.0
    flip_prob = flip_prob
    coeffs = 1/torch.linspace(1,D,D)[:,None]**1
    coeffs *= 1/coeffs.sum()
    means = torch.tensor([[-2, 0]]).T # means for mixture U dist
    scales = torch.tensor([[-1.0, 1.0]]).T  # variances for mixture U dist
    probabilities = torch.tensor([1/2,1/2]) # mixture probs for mixture U dist
    base_dists = [HalfNormal(1),HalfCauchy(1)]
    noise_dist = Mixture1D(base_dists,probabilities,means,scales)
    Zdist = Normal(0,1.5)

    # Feature (CDF) set-up
    feature = lambda x,t: (torch.sigmoid(x)[None]<= t[...,None]).float() # N X M -> D x N x M
    t = torch.linspace(0,1,1000)[:,None]
    
    # Method + opt set up
    
    # Cocycle training and CV
    cocycle_loss = "CMMD_U"
    med_heuristic_samples = 10**4
    validation_method = "CV"
    choose_best_model = "overall"
    cocycle = cocycle
    retrain = True
    layers = 2
    width = width
    train_val_split = 0.8
    folds = int(1/(1-train_val_split))
    batch_size = 64
    learn_rate = [1e-3]
    scheduler = True
    maxiter = 10000
    miniter = 10000
    weight_decay = weight_decay
    RQS_bins = 8
    val_batch_size = 1024
    
    # Setting training optimiser args
    opt_args = ["learn_rate",
                "scheduler",
                "batch_size",
                "maxiter",
                "miniter",
                "weight_decay",
                "print_",
                "val_batch_size"]
    opt_argvals = [learn_rate,
                  scheduler,
                  batch_size,
                 maxiter,
                  miniter,
                  weight_decay,
                  False,
                  val_batch_size]
    
    hyper = ["weight_decay"]
    hyper_val = [weight_decay]
    
    #Shorthand function calls
    def NN(i,o=1,width=128,layers=2):
        return NN_RELU_Conditioner(width = width,
                                         layers = layers, 
                                         input_dims =  i, 
                                         output_dims = o,
                                         bias = True)
    
    
    # Storage objects
    ATE_PI_term = torch.zeros((splits,len(t)))
    ATE_IPW = torch.zeros((splits,len(t)))
    ATE_DR = torch.zeros((splits,len(t)))
    
    weights1,weights0 = [],[]

    # Getting data
    torch.manual_seed(seed)
    Z,X,Y = DGP(N,D,partial(policy,flip_prob = flip_prob),coeffs,Zcorr,Zdist,noise_dist)
    torch.manual_seed(seed)
    Zint,Xint,Yint = DGP(Nint,D,partial(new_policy),coeffs,Zcorr,Zdist,noise_dist)
    
    # Adjusting dimensions
    D += 1
    P = D - 1
    
    # Getting sample splits for DR estimation
    Xsplits = get_CV_splits(X,splits)
    Xintsplits = get_CV_splits(Xint[:N],splits)
    Ysplits = get_CV_splits(Y,splits)
    
    # Specifying models for cross-validation
    if cocycle == "linear":
        conditioners_list = [[Lin_Conditioner(D,1)]]
        transformers_list = [Transformer([Shift_layer()])]
    if cocycle == "additive":
        conditioners_list = [[NN(D,1,width,layers)]]
        transformers_list = [Transformer([Shift_layer()])]
    if cocycle == "affine":
        conditioners_list = [[NN(D,1,width,layers),NN(D,1,width,layers)]]
        transformers_list = [Transformer([Shift_layer(),Scale_layer()])]
    if cocycle == "TMI":
        conditioners_list = [[NN(D,1,width,layers),NN(D,1,width,layers),NN(D,3*RQS_bins+2,width,layers)]]
        transformers_list = [Transformer([Shift_layer(),Scale_layer(),RQS_layer(RQS_bins)])]
        
    models_validation = []
    for m in range(len(conditioners_list)):
        models_validation.append(cocycle_model(conditioners_list[m],transformers_list[m]))
    hyper_args = [hyper]*len(conditioners_list)
    hyper_argvals = [hyper_val]*len(conditioners_list)
    
    # Getting loss functon (using CMMD_V as scalable for validation)
    loss_fn =  Loss(loss_fn = cocycle_loss,kernel = [gaussian_kernel(torch.ones(1),1)]*2)
    loss_fn_val =  Loss(loss_fn = "CMMD_V",kernel = [gaussian_kernel(torch.ones(1),1)]*2)
    loss_fn.median_heuristic(X,Y,subsamples = med_heuristic_samples)
    loss_fn_val.median_heuristic(X,Y,subsamples = med_heuristic_samples)    
    
    # Getting plug-in estimator
    final_model_PI,val_losses_PI = validate(models_validation,
                                         loss_fn,
                                         X,
                                         Y,
                                         loss_fn_val,
                                         validation_method,
                                         train_val_split,
                                         opt_args,
                                         opt_argvals,
                                         hyper_args,
                                         hyper_argvals,
                                         choose_best_model,
                                         retrain)
    
    # Getting potential outcomes and plug-in estimator
    Yintpred = final_model_PI.cocycle(Xint[:N],X,Y).detach()
    ATE_PI = feature(Yintpred,t).mean((1,2))
        
    # Getting DR estimator
    val_losses_DR = []
    for k in range(splits):

        # Getting dataset
        Xtrain,Ytrain = Xsplits[k][0],Ysplits[k][0]
        Xtest,Ytest = Xsplits[k][1],Ysplits[k][1]
        Xinttest = Xintsplits[k][1]

        # Getting model for kth split
        final_model_DR,val_losses = validate(models_validation,
                                         loss_fn,
                                         Xtrain,
                                         Ytrain,
                                         loss_fn_val,
                                         validation_method,
                                         train_val_split,
                                         opt_args,
                                         opt_argvals,
                                         hyper_args,
                                         hyper_argvals,
                                         choose_best_model,
                                         retrain)
        val_losses_DR.append(val_losses)

        # Getting potential outcomes and plug-in estimator
        batch = 100
        nbatch = int(len(t)/batch)
        outer_preds = final_model_DR.cocycle_outer(Xinttest,Xtrain,Ytrain).detach()
        for i in range(nbatch):
            ATE_PI_term[k,i*batch:(i+1)*batch] = feature(outer_preds,
                                                    t[i*batch:(i+1)*batch]).mean((1,2))
            
            # Estimating propensity model(s)
            Propensity_score_model_est = []
            Propensity_score_model_policy = []
            Propensity_score_model_new_policy = []
            Xtrue = policy(Xtrain[:,1:])
            states = torch.unique(Xtrain[:,0]).int()
            nstate = len(states)
            P = torch.zeros((nstate,nstate))
            for i in range(nstate):
                for j in range(nstate):
                    P[i,j] = ((Xtrain[:,0]==states[i])*(Xtrue[:,0]==states[j])).float().sum()
            P *= 1/P.sum(0)
            propensity_model_est = propensity_score(P,policy)
            propensity_model_new_policy = propensity_score(torch.eye(len(P)),new_policy)  
            propensity_model_policy = propensity_score(torch.eye(len(P)),policy) 
    
            # Getting IPW weights and estimator
            weights_int = (propensity_model_new_policy(Xtest,Xtest[:,1:])/
                            propensity_model_est(Xtest,Xtest[:,1:]).detach())
    
            ATE_IPW[k] = (weights_int[None,:,None]*feature(Ytest,t)).mean((1,2))
    
            # Getting conditional expectations and DR estimator
            ATE_DR[k] = ATE_PI_term[k]+ ATE_IPW[k]
            outer_preds = final_model_DR.cocycle_outer(Xtest,Xtrain,Ytrain).detach()
            for i in range(nbatch):
                ATE_DR[k,i*batch:(i+1)*batch] -= (weights_int[None,:]*
                                                  feature(outer_preds,
                                                          t[i*batch:(i+1)*batch]).mean(-1)
                                                  ).mean(1)
   
    # Getting true CDF
    true_cdf = feature(Yint,t).mean((1,2))

    results = {"name": name,
               "ATE_PI" : ATE_PI,
               "ATE_PI_term" : ATE_PI_term,
               "ATE_IPW" : ATE_IPW,
               "ATE_DR" : ATE_DR,
               "val_loss_PI" : val_losses_PI,
               "val_loss_DR" : val_losses_DR}

    return results