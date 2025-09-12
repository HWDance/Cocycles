#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Imports
import torch
from torch.distributions import Normal,Gamma,HalfNormal,HalfCauchy
import os
import time
from functools import partial

from causal_cocycle.model import cocycle_model,flow_model,cocycle_outcome_model
from causal_cocycle.optimise import *
from causal_cocycle.loss_functions import Loss
from causal_cocycle.conditioners import Empty_Conditioner,Constant_Conditioner,Lin_Conditioner,NN_RELU_Conditioner
from causal_cocycle.transformers import Transformer,Shift_layer,Scale_layer,RQS_layer
from causal_cocycle.helper_functions import likelihood_loss,mmd,propensity_score
from causal_cocycle.kernels import *
from causal_cocycle.kde import *

from BD import *

#Shorthand function calls
def NN(i,o=2,width=128,layers=2):
    return NN_RELU_Conditioner(width = width,
                                     layers = layers, 
                                     input_dims =  i, 
                                     output_dims = o,
                                     bias = True)

T = partial(Transformer,logdet = True)

# DGP set up
N = 5000
D = 10
Zcorr = 0.0
flip_prob = 0.05
coeffs = 1/torch.linspace(1,D,D)[:,None]**1
coeffs *= 1/coeffs.sum()
means = torch.tensor([[-2, 0]]).T # means for mixture U dist
scales = torch.tensor([[-1.0, 1.0]]).T  # variances for mixture U dist
probabilities = torch.tensor([1/2,1/2]) # mixture probs for mixture U dist
#base_dists = [IG(10,10),IG(1,1)]
base_dists = [HalfNormal(1),HalfCauchy(1)]
noise_dist = Mixture1D(base_dists,probabilities,means,scales)
Zdist1 = Normal(0,1.5)
Zdist2 = Normal(0,1.5)
feature = lambda x,t: (torch.sigmoid(x)<=t.T).float()
nintsample = 10**4
t = torch.linspace(0,1,1000)[:,None]

# Method + opt set up
cocycle_loss = "CMMD_U"
batch_size =64
validation_method = "CV"
choose_best_model = "per_fold"
layers = 2
width = 128
train_val_split = 0.5
learn_rate = [1e-3]
scheduler = True
maxiter = 10000
miniter = 10000
weight_decay = 1e-3
RQS_bins = 8
val_batch_size = N

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
    
def run_experiment(seed,hyper,hyper_val,flip_prob,N):
    
    # Specifying models for cross-validation
    conditioners_list = [[NN(D+1,1,width,layers)],
                              [NN(D+1,1,width,layers),NN(D+1,1,width,layers)],
                              [NN(D+1,1,width,layers),NN(D+1,1,width,layers),NN(D+1,3*RQS_bins + 2,width,layers)]]
    transformers_list = [Transformer([Shift_layer()]),
                               Transformer([Shift_layer(),Scale_layer()]),
                               Transformer([Shift_layer(),Scale_layer(),RQS_layer(RQS_bins)])]
    models_validation = []
    for m in range(len(conditioners_list)):
        models_validation.append(flow_model(conditioners_list[m],transformers_list[m]))
    hyper_args = [hyper]*len(conditioners_list)
    hyper_argvals = [hyper_val]*len(conditioners_list)
    
    # DGP
    torch.manual_seed(seed)
    Z1,X1,Y1 = DGP(N,D,partial(policy,flip_prob = flip_prob),coeffs,Zcorr,Zdist1,noise_dist)
    Z2,X2,Y2 = DGP(N,D,partial(policy,flip_prob = flip_prob),coeffs,Zcorr,Zdist2,noise_dist)
    Z,X,Y = (torch.row_stack((Z1,Z2)),
             torch.row_stack((X1,X2)),
             torch.row_stack((Y1,Y2)))
    
    # Getting loss functon (using CMMD_V as scalable for validation)
    loss_fn =  Loss(loss_fn = cocycle_loss,kernel = [gaussian_kernel(torch.ones(1),1)]*2)
    loss_fn_val =  Loss(loss_fn = "CMMD_V",kernel = [gaussian_kernel(torch.ones(1),1)]*2)
    loss_fn.median_heuristic(X1,Y1,subsamples = 10**4)
    loss_fn_val.median_heuristic(X1,Y1,subsamples = 10**4)

    # Cross-validation
    final_models,val_losses = validate(models_validation,
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
                                         choose_best_model)
    
    # Defining outcome model and feature of interest (i.e. cdf)
    feature = lambda x,t: (torch.sigmoid(x)[None]<= t[...,None]).float()
    conditional_mean_model1 = cocycle_outcome_model(final_models[0],X2,Y2)
    conditional_mean_model2 = cocycle_outcome_model(final_models[1],X1,Y1)

    # Sampling from interventional distribution
    ninttrain = N
    torch.manual_seed(seed)
    Zint1,Xint1,Yint1 = DGP(int(nintsample/2),D,partial(new_policy),coeffs,Zcorr,Zdist1,noise_dist)
    Zint2,Xint2,Yint2 = DGP(int(nintsample/2),D,partial(new_policy),coeffs,Zcorr,Zdist2,noise_dist)
   
    # cdf values
    
    # Cocycle model cdf
    batch = 10
    nbatch = int(len(t)/batch)
    cocycle_cdf_int1 = torch.zeros(len(t))
    cocycle_cdf_int2 = torch.zeros(len(t))
    for i in range(nbatch):
        cocycle_cdf_int1[i*batch:(i+1)*batch] = conditional_mean_model1(Xint1[:ninttrain],
                                                    partial(feature,t = t[i*batch:(i+1)*batch])).detach().mean(1)
        cocycle_cdf_int2[i*batch:(i+1)*batch] = conditional_mean_model2(Xint2[:ninttrain],
                                                    partial(feature,t = t[i*batch:(i+1)*batch])).detach().mean(1)
            
    # True cdf
    true_cdf_int1 = feature(Yint1,t).mean((1,2))
    true_cdf_int2 = feature(Yint2,t).mean((1,2))
    
    # Training propensity score models
    Propensity_score_model_est = []
    Propensity_score_model_policy = []
    Propensity_score_model_new_policy = []
    
    # Estimating mistake probabilities
    Xtrue = policy(Z)
    states = torch.unique(X[:,0]).int()
    nstate = len(states)
    P = torch.zeros((nstate,nstate))
    for i in range(nstate):
        for j in range(nstate):
            P[i,j] = ((X[:,0]==states[i])*(Xtrue[:,0]==states[j])).float().sum()
    P *= 1/P.sum(0)
    
    propensity_model_est = propensity_score(P,policy)
    propensity_model_new_policy = propensity_score(torch.eye(len(P)),new_policy)  
    propensity_model_policy = propensity_score(torch.eye(len(P)),policy) 
    
    # Getting IPW estimator
    weights_int1 = (propensity_model_new_policy(X1,Z1)/
                    propensity_model_est(X1,Z1)).detach()
    
    weights_int2 = (propensity_model_new_policy(X2,Z2)/
                    propensity_model_est(X2,Z2)).detach()
    
    #weights_int1 *= len(weights_int1)/weights_int1.sum()
    #weights_int2 *= len(weights_int2)/weights_int2.sum()
    
    IPW_cdf_int1 = (weights_int1[None,:,None]*feature(Y1,t)).mean((1,2))
    IPW_cdf_int2 = (weights_int2[None,:,None]*feature(Y2,t)).mean((1,2))

    # Getting DR estimator (start by adding on IPW term to outcome model
    cocycle_DR_cdf_int1 = cocycle_cdf_int1 + IPW_cdf_int1
    cocycle_DR_cdf_int2 = cocycle_cdf_int2 + IPW_cdf_int2
    
    for i in range(nbatch):
        # Getting batch of conditional means and propensity weights
        conditional_mean_batch1 = conditional_mean_model1(X1,partial(feature,t = t[i*batch:(i+1)*batch])).detach()
        conditional_mean_batch2 = conditional_mean_model2(X2,partial(feature,t = t[i*batch:(i+1)*batch])).detach()
    
        # Updating DR estimator
        cocycle_DR_cdf_int1[i*batch:(i+1)*batch] -= (weights_int1*conditional_mean_batch1).mean(1)
        cocycle_DR_cdf_int2[i*batch:(i+1)*batch] -= (weights_int2*conditional_mean_batch2).mean(1)
            
    # Returning trial
    return {"name": "cocycles_{0}={1}".format(str(hyper),str(hyper_val)),
            "val_loss" : val_losses.min(0)[0],
            "plug-in_1": cocycle_cdf_int1,
            "plug-in_2": cocycle_cdf_int2,
            "DR-1": cocycle_DR_cdf_int1,
            "DR-2": cocycle_DR_cdf_int2,
            "IPW-1": IPW_cdf_int1,
            "IPW-2": IPW_cdf_int2,
            "true-1": true_cdf_int1,
            "true-2": true_cdf_int1,
            "seed": seed}




