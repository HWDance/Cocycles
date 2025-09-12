from torch.distributions import Normal,Laplace,Uniform,Gamma,Cauchy, Beta
import os
import time

from causal_cocycle.model import cocycle_model,flow_model
from causal_cocycle.optimise import *
from causal_cocycle.loss_functions import Loss
from causal_cocycle.conditioners import Lin_Conditioner
from causal_cocycle.transformers import Transformer, Shift_layer
from causal_cocycle.helper_functions import likelihood_loss
from causal_cocycle.kernels import *

"""
Function to run regression estimation experiment
"""
def run_experiment(seed,N):

    """
    Configs
    """
    # Experimental set up
    N,D,P = N,1,1
    sig_noise_ratio = 1
    
    # Training set up
    train_val_split = 1
    ntrain = int(train_val_split*N)
    learn_rate = [1e-2]
    scheduler = True
    val_tol = 1e-3
    batch_size = 128
    val_loss = False
    maxiter = 3000
    miniter = 3000
    bias = False

    """
    Main
    """
    # Object storage
    names = ["L2","L1","HSIC","URR","CMMD-V","CMMD-U","True"]
    Coeffs = torch.zeros((1,len(names),P))
    
    # Data generation

    # Drawing data
    torch.manual_seed(seed)
    X = Normal(1,1).sample((N,D))
    X *= 1/(D)**0.5
    B = torch.ones((D,1))*(torch.linspace(0,D-1,D)<P)[:,None]
    F = X @ B
    e = Beta(0.5,0.5).sample((N,1))
    e2 = Beta(0.5,0.5).sample((N,1))
    U = (e/(1-e))*(e2/(1-e2))*Normal(0,1).sample((N,1))/sig_noise_ratio**0.5
    Y = F + U

    # Training with L2
    if bias:
        Xtild = torch.column_stack((torch.ones((N,1)),X))
    else:
        Xtild = X
    LS_model = torch.linalg.solve(Xtild.T @ Xtild, Xtild.T @ Y)[(bias*1):]
    # Training with L2
    if bias:
        Xtild = torch.column_stack((torch.ones((N,1)),X))
    else:
        Xtild = X
    LS_model = torch.linalg.solve(Xtild.T @ Xtild, Xtild.T @ Y)[(bias*1):]

    # Training with L1
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = likelihood_loss(Laplace(0,1), log_det = False)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    L1_model = cocycle_model([conditioner],transformer)
    optimise(L1_model,loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = N,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    # Training with HSIC (Median-heuristic L2)
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "HSIC",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,Y-X @ LS_model, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    HSICL2_model = cocycle_model([conditioner],transformer)
    optimise(HSICL2_model,loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    # Training with CMMD
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "CMMD_V",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,Y, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    CMMD_model = cocycle_model([conditioner],transformer)
    optimise(CMMD_model,loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    # Training with CMMD (unbiased implementation)
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "CMMD_U",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,Y, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    CMMDU_model = cocycle_model([conditioner],transformer)
    optimise(CMMDU_model,loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    
    # Training with URR (1:1 sample ratio)
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "URR",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,Y, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    URR_model = cocycle_model([conditioner],transformer)
    optimise(URR_model,loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    # Storing results
    Coeffs[:,0] = LS_model.T
    Coeffs[:,1] = L1_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[:,2] = HSICL2_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[:,3] = URR_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[:,4] = CMMD_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[:,5] = CMMDU_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[:,6] = B.T
    
    # Saving output
    return {"seed" : seed,
            "names": names, 
            "Coeffs": Coeffs,
            "dist" : "Cauchy",
            "nsamples" : N}