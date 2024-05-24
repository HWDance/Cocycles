#!/usr/bin/env python
# coding: utf-8

# In[40]:

# Imports
import torch
from torch.distributions import Normal,Gamma,HalfNormal,HalfCauchy
import os
import time
from functools import partial

from causal_cocycle.kernels import *
from causal_cocycle.kde import *
from causal_cocycle.regression_functionals import *
from causal_cocycle.distribution_estimation import *
from causal_cocycle.helper_functions import propensity_score

from BD import *


# In[100]:

def run_experiment(seed,flip_prob,N):
    
    # DGP set up
    N = N
    D = 10
    Zcorr = 0.0
    flip_prob = flip_prob
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
    # Method + opt set up
    functional = KRR_functional
    kernel = gaussian_kernel
    subsample = False
    subsamples = N
    nfold = 5
    ls_method = "med_heuristic"
    hyper_lambda = 2**torch.linspace(-10,0,5)
    hyper_ls = 2**torch.linspace(-1,1,5)
    
    hyper_grid_lambda = hyper_lambda.repeat(len(hyper_ls))
    hyper_grid_ls = torch.repeat_interleave(hyper_ls,len(hyper_lambda))    
    t_train = torch.linspace(0,1,1000)[:,None]
    
    # DGP
    torch.manual_seed(seed)
    Z1,X1,Y1 = DGP(N,D,partial(policy,flip_prob = flip_prob),coeffs,Zcorr,Zdist1,noise_dist)
    Z2,X2,Y2 = DGP(N,D,partial(policy,flip_prob = flip_prob),coeffs,Zcorr,Zdist2,noise_dist)
    Z,X,Y = (torch.row_stack((Z1,Z2)),
             torch.row_stack((X1,X2)),
             torch.row_stack((Y1,Y2)))
    
    # Getting median heuristic lengthscale
    Distances_1= torch.tril((X1[...,None]-X1[...,None].T)**2)
    Distances_2= torch.tril((X2[...,None]-X2[...,None].T)**2)
    ls_1 = torch.ones(D+1)
    ls_2 = torch.ones(D+1)
    for d in range(D+1):
        ls_1[d] = (Distances_1[:,d][Distances_1[:,d]!=0].median()/2).sqrt()
        ls_2[d] = (Distances_2[:,d][Distances_2[:,d]!=0].median()/2).sqrt()

    # Getting hypergrids
    hyper_grid1,hyper_grid2 = [],[]
    for h in range(len(hyper_grid_lambda)):
        hyper_grid1.append([hyper_grid_ls[h]*ls_1,hyper_grid_lambda[h]])
        hyper_grid2.append([hyper_grid_ls[h]*ls_2,hyper_grid_lambda[h]])
      
                          
    # Defining model
    kx_1 = kernel(lengthscale = ls_1)
    kx_2 = kernel(lengthscale = ls_2)
    regressor_1 = functional(kx_1)
    regressor_2 = functional(kx_2)
    CE1 = Conditional_Expectation_Regressor(regressor_1)
    CE2 = Conditional_Expectation_Regressor(regressor_2)
    
    # Doing CV over hypers
    CE1.CVgridsearch(X1,feature(Y1,t_train).float(),
                            nfold = nfold, 
                            subsample = subsample,
                            subsamples = subsamples,
                            hyper_grid = hyper_grid1)
    CE2.CVgridsearch(X2,feature(Y2,t_train).float(),
                            nfold = nfold, 
                            subsample = subsample,
                            subsamples = subsamples,
                            hyper_grid = hyper_grid2)
    
    # Sampling from interventional distribution
    nintsample = 10**4
    ninttrain = N
    torch.manual_seed(seed)
    Zint1,Xint1,Yint1 = DGP(int(nintsample/2),D,partial(new_policy),coeffs,Zcorr,Zdist1,noise_dist)
    Zint2,Xint2,Yint2 = DGP(int(nintsample/2),D,partial(new_policy),coeffs,Zcorr,Zdist2,noise_dist)
   
    # cdf values
    t = torch.linspace(0,1,1000)[:,None]
    
    # Getting CDF
    kde_cdf_int1 = CE2.forward(feature(Y2,t).float(),X2,Xint1[:ninttrain]).mean(0).detach()
    kde_cdf_int2 = CE1.forward(feature(Y1,t).float(),X1,Xint2[:ninttrain]).mean(0).detach()
    
    # True cdf
    true_cdf_int1 = feature(Yint1,t).mean(0)
    true_cdf_int2 = feature(Yint2,t).mean(0)

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
    
    IPW_cdf_int1 = (weights_int1[:,None]*feature(Y1,t)).mean(0)
    IPW_cdf_int2 = (weights_int2[:,None]*feature(Y2,t)).mean(0)
    
    # Getting DR estimator (start by adding on IPW term to outcome model
    kde_DR_cdf_int1 = kde_cdf_int1 + IPW_cdf_int1
    kde_DR_cdf_int2 = kde_cdf_int2 + IPW_cdf_int2
    
    # Updating DR estimator
    kde_DR_cdf_int1 -= (weights_int1[:,None]*CE2.forward(feature(Y2,t).float(),X2,X1)).mean(0).detach()
    kde_DR_cdf_int2-= (weights_int2[:,None]*CE1.forward(feature(Y1,t).float(),X1,X2)).mean(0).detach()
    
    # Returning trial
    return {"name": "RKHS",
            "plug-in_1": kde_cdf_int1,
            "plug-in_2": kde_cdf_int2,
            "DR-1": kde_DR_cdf_int1,
            "DR-2": kde_DR_cdf_int2,
            "IPW-1": IPW_cdf_int1,
            "IPW-2": IPW_cdf_int2,
            "true-1": true_cdf_int1,
            "true-2": true_cdf_int1,
            "seed": seed}




