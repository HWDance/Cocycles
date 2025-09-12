#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Imports
import torch
from torch.distributions import Normal,Gamma,HalfNormal,HalfCauchy
import os
import time
from functools import partial

from econml.dml import CausalForestDML
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
batch_size =128
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
    
def run_experiment(seed,hyper,hyper_val,flip_prob,N):
    

    
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
    feature = lambda x,t: (torch.sigmoid(x)<= t.T).float()
    est = CausalForestDML(discrete_treatment=True)
    est.fit(feature(Y,t), T=X[:,0], X=Z, W=None)
    
    # Sampling from interventional distribution
    ninttrain = N
    torch.manual_seed(seed)
    Zint1,Xint1,Yint1 = DGP(int(nintsample/2),D,partial(new_policy),coeffs,Zcorr,Zdist1,noise_dist)
    Zint2,Xint2,Yint2 = DGP(int(nintsample/2),D,partial(new_policy),coeffs,Zcorr,Zdist2,noise_dist)
    
    # Getting CDF-predidctions
    
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




