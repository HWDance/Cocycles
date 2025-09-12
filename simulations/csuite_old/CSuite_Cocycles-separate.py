
# Imports
import torch
from torch.distributions import Normal,Uniform,Gamma,Laplace
import os
import time

from causal_cocycle.model import cocycle_model,flow_model
from causal_cocycle.optimise import *
from causal_cocycle.loss_functions import Loss
from causal_cocycle.conditioners import Empty_Conditioner,Constant_Conditioner,Lin_Conditioner,NN_Tanh_Conditioner
from causal_cocycle.transformers import Transformer,Shift_layer,Scale_layer,RQS_layer,Inverse_layer
from causal_cocycle.helper_functions import likelihood_loss,mmd
from causal_cocycle.kernels import *
from causal_cocycle.CDAGM import *

#Shorthand function calls
def NN(i,width,layers):
    return NN_Tanh_Conditioner(width = width,
                                     layers = layers, 
                                     input_dims =  i, 
                                     output_dims = 1,
                                     bias = True)

def run_experiment(seed,dataset,base_dist,base_transform,num_base_change,cocycle_loss,batch_size,scheduler,weight_decay):

    """
    Configs
    """
    # Experimental settings
    n,nint = 5000,10**6
    int_levels = [-2,-1,0,1,2]
    intervention = lambda a,x : a+x*0
    
    # Method + opt set up
    validation_method = "fixed"
    choose_best_model = "overall"
    retrain = False
    layers = 2
    widths = [32,64,128]
    train_val_split = 0.8
    learn_rate = [1e-3]
    scheduler = True
    maxiter = 10000
    miniter = 10000
    bins = 8

    # MMD estimation set up
    mmd_samples = 10**4
    heuristic_samples = 10**3

    """
    Model + dataset set up
    """
    
    # Naming method
    model_name = "Cocycles_{0}".format(cocycle_loss)

    # Setting causal graph
    if dataset == "Fork":
        from CSuite import Fork_Nonlin as DGP
        from CSuite import Fork_conditioner as DGP_info
    elif dataset == "Simpson":
        from CSuite import Simpson_Nonlin as DGP
        from CSuite import Simpson_conditioner as DGP_info
    else:
        from CSuite import Nonlin_Gauss_dense as DGP
        from CSuite import NonlinGaussdense_conditioner as DGP_info
    parents = DGP_info().parents
    
    # Setting training optimiser args
    opt_args = ["learn_rate",
                "scheduler",
                "batch_size",
                "maxiter",
                "miniter",
                "weight_decay"]
    opt_argvals = [learn_rate,
                  scheduler,
                  batch_size,
                  maxiter,
                  miniter,
                  weight_decay]

    # Storage objects
    ATE = torch.zeros((1,len(int_levels)))
    E_DO = torch.zeros((1,len(int_levels)))
    Training_time = torch.zeros((1,1))
    MMD = torch.zeros((1,len(int_levels)))
    models = []

    # Drawing data
    torch.manual_seed(seed)
    noise_dists = [base_dist]*num_base_change + [Normal(0,1)]*(len(parents)-num_base_change)
    noise_dist_transforms = [base_transform]*num_base_change + [lambda x : x]*(len(parents)-num_base_change)
    Xobs,Xint = DGP(n, nint, True, intervention, int_levels, noise_dists,noise_dist_transforms)
    Xobs,Xobstest = Xobs[:n],Xobs[n:]

    
    # Estim/ating models
    start_time = time.time()
    for i in range(len(parents)):
        if parents[i]:
            # Getting nodes from graph
            index_x,index_y = parents[i],[i]
            inputs,outputs = Xobs[:,index_x].view(n,len(index_x)),Xobs[:,index_y].view(n,len(index_y))
    
            # Specifying models for cross-validation
            conditioners_validation = ([[NN(len(parents[i]),widths[m],layers)] for m in range(len(widths))]+
                                      [[NN(len(parents[i]),widths[m],layers),
                                       NN(len(parents[i]),widths[m],layers)] for m in range(len(widths))]+
                                      [[NN(len(parents[i]),widths[m],layers),
                                      NN(len(parents[i]),widths[m],layers),
                                      NN(len(parents[i]),widths[m],layers)] for m in range(len(widths))])
            transformers_validation = ([Transformer([Shift_layer()])]*len(widths)+
                                      [Transformer([Shift_layer(),Scale_layer()])]*len(widths)+
                                      [Transformer([Shift_layer(),Scale_layer(),RQS_layer(bins=8)])]*len(widths))
            models_validation = [cocycle_model(conditioners_validation[m],
                                       transformers_validation[m]) for m in range(len(widths))]
            kernel = gaussian_kernel(1,1)
            loss_fn =  Loss(loss_fn = cocycle_loss,kernel = [kernel]*2)
            loss_fn.median_heuristic(inputs,outputs, subsamples = 10**4)                        
            final_model,val_losses = validate(models_validation,
                                         loss_fn,
                                         inputs,
                                         outputs,
                                         loss_fn,
                                         validation_method,
                                         train_val_split,
                                         opt_args,
                                         opt_argvals,
                                         choose_best_model = choose_best_model,
                                         retrain = retrain)
            models.append(final_model[0])
        else:
            models.append([])
    Training_time[:,0] = time.time()-start_time
    
    # Defining CCDAGM model
    ccdagm = CCDAGM(models,parents)
    
    # Interventional prediction
    for i in range(len(int_levels)):
        
        # Setting intervention level
        a = int_levels[i]
        
        # Getting interventional_samples
        Xpred,Xintpred = ccdagm.interventional_dist_sample(Xobs,
                                                     intervention,
                                                     ["id",a,"id","id"],
                                                     len(Xobs),
                                                     uniform_subsample = False)
        # Getting ATEs                      
        E_DO[:,i] = Xintpred[:,-1].detach().mean() - Xint[i][:,-1].mean()
        ATE[:,i] = E_DO[:,i] - (Xpred[:,-1].detach().mean()-Xobstest[:,-1].mean())
        
        # Getting MMD
        Ypred,Y = Xintpred[:,-1:],Xint[i][:,-1:]
        MMD[:,i] = mmd(kernel = gaussian_kernel(1,1))(Y,Ypred,
            median_heuristic = True,
            mmd_samples = mmd_samples,
            heuristic_samples = heuristic_samples)                                 

    return { "names": model_name,
            "dataset": dataset,
            "base dist": base_dist,
            "EY|DO(X)": E_DO,
            "ATE": ATE,
            "MMD": MMD,
            "training time": Training_time,
            "seed":seed
           }
        

    