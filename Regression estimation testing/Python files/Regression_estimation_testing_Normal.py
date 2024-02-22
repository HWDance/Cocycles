# User
user = "nk1922"

"""
Imports
"""
# Librairies
import torch
from torch import nn
from torch.distributions import Normal,Laplace,Uniform,Gamma
import matplotlib.pyplot as plt
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Cocycles code
os.chdir('C:/Users/{0}/OneDrive/Documents/Cocycles project/Cocycle_code'.format(user))
from Cocycle_CDAGM import *
from Cocycle_model import *
from Cocycle_optimise import *
from Cocycle_loss_functions import *
from Conditioners import *
from Transformers import *
from KDE_estimation import *
from Kernels import *
from Helper_functions import *
os.chdir('C:/Users/{0}/OneDrive/Documents/Cocycles project/Experiments_code'.format(user))

"""
Configs
"""
# Experimental set up
trials = 50
N,D,P = 200,1,1
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
names = ["L2","L1","HSIC_Y","HSIC_L2","HSIC_L1","URR","CMMD","CMMD_unbiased","True"]
Coeffs = torch.zeros((len(names),trials,P))

# Data generation
for t in range(trials):
    
    # Drawing data
    torch.manual_seed(t)
    X = Normal(1,1).sample((N,D))
    X *= 1/(D)**0.5
    B = torch.ones((D,1))*(torch.linspace(0,D-1,D)<P)[:,None]
    F = X @ B
    U = Normal(0,1).sample((N,1))/sig_noise_ratio
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
    L1_model = Train(L1_model).optimise(loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = N,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)

    # Training with HSIC (Median-heuristic Y)
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "HSIC",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,Y, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    HSICY_model = cocycle_model([conditioner],transformer)
    HSICY_model = Train(HSICY_model).optimise(loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    # Training with HSIC (Median-heuristic L2)
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "HSIC",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,Y-X @ LS_model, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    HSICL2_model = cocycle_model([conditioner],transformer)
    HSICL2_model = Train(HSICL2_model).optimise(loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    # Training with HSIC (Median-heuristic L1)
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "HSIC",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,L1_model.inverse_transformation(X,Y).detach(), subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    HSICL1_model = cocycle_model([conditioner],transformer)
    HSICL1_model = Train(HSICL1_model).optimise(loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)

    # Training with CMMD
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "CMMD_M",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,Y, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    CMMD_model = cocycle_model([conditioner],transformer)
    CMMD_model = Train(CMMD_model).optimise(loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    # Training with CMMD (unbiased implementation)
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "CMMD_U",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)],get_CMMD_mask = True, mask_size = batch_size)
    loss_fn.median_heuristic(X,Y, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    CMMDU_model = cocycle_model([conditioner],transformer)
    CMMDU_model = Train(CMMDU_model).optimise(loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    
    # Training with URR (1:1 sample ratio)
    inputs_train,outputs_train, inputs_val,outputs_val  = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]
    loss_fn = Loss(loss_fn = "URR",kernel = [gaussian_kernel(torch.ones(1),1),gaussian_kernel(torch.ones(1),1)])
    loss_fn.median_heuristic(X,Y, subsamples = 10**4)
    conditioner = Lin_Conditioner(D,1, bias = bias)
    transformer = Transformer([Shift_layer()])
    URR_model = cocycle_model([conditioner],transformer)
    URR_model = Train(URR_model).optimise(loss_fn,inputs_train,outputs_train,inputs_val,outputs_val, batch_size = batch_size,learn_rate = learn_rate,
                                         print_ = True,plot = False, miniter = miniter,maxiter = maxiter, val_tol = val_tol,val_loss = val_loss,
                                 scheduler = scheduler)
    
    # Storing results
    Coeffs[0,t] = LS_model.T
    Coeffs[1,t] = L1_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[2,t] = HSICY_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[3,t] = HSICL2_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[4,t] = HSICL1_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[5,t] = URR_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[6,t] = CMMD_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[7,t] = CMMDU_model.conditioner[0].state_dict()['stack.0.weight']
    Coeffs[8,t] = B.T
    
    # Saving output
    os.chdir('C:/Users/{0}/OneDrive/Documents/Cocycles project'.format(user))
    torch.save({ "names": names, 
                "Coeffs": Coeffs},
               f = f'Experimental_results/'+'Regression_estimation_testing_normal_N={0}_D={1}_P={2}_snr={3}_trials={4}.pt'.format(N,D,P,sig_noise_ratio,trials)
              )