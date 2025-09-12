
# Imports
import torch
from torch.distributions import Normal,Uniform,Gamma,Laplace,StudentT
import os
import time
import copy

from causal_cocycle.model import cocycle_model,flow_model
from causal_cocycle.optimise import *
from causal_cocycle.loss_functions import Loss
from causal_cocycle.conditioners import Empty_Conditioner,Constant_Conditioner,Lin_Conditioner,NN_Tanh_Conditioner
from causal_cocycle.transformers import Transformer,Shift_layer,Scale_layer,RQS_layer,Inverse_layer
from causal_cocycle.helper_functions import likelihood_loss,mmd, SCM_intervention_sample, kolmogorov_distance
from causal_cocycle.kernels import *
from causal_cocycle.CDAGM import *

#Shorthand function calls
def NN(i,width,layers,p=1):
    if i>=1:
        return NN_Tanh_Conditioner(width = width,
                                     layers = layers, 
                                     input_dims =  i, 
                                     output_dims = p,
                                     bias = True)
    else:
        return C(1,1,1)
def C(rows,cols,value):
    return Constant_Conditioner(init = torch.ones((rows,cols))*value)


def run_experiment(seed,
                   dataset,
                   base_dist,
                   base_transform,
                   num_base_change,
                   model_name,
                   model_base_dist,
                   use_RQS_base_flow,
                   RQS_bins,
                   tail_adapt,
                   batch_size,
                   scheduler,
                   weight_decay):

    """
    Configs
    """
    # Experimental settings
    n,nint = 5000,10**6
    int_levels = [-2,-1,0,1,2]
    intervention = lambda a,x : a+x*0
    mc_samples = 10**5
    
    # Method + opt set up
    loss = likelihood_loss
    validation_method = "fixed"
    choose_best_model = "overall"
    retrain = False
    layers = 2
    widths = [32,64,128]
    train_val_split = 0.8
    learn_rate = [1e-3]
    maxiter = 10000
    miniter = 10000

    # MMD estimation set up
    mmd_samples = 10**4
    heuristic_samples = 10**3

    """
    Model set up
    """
    
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
                "weight_decay",
                "likelihood_param_opt"]
    opt_argvals = [learn_rate,
                  scheduler,
                  batch_size,
                  maxiter,
                  miniter,
                  weight_decay,
                  tail_adapt]

    # Storage objects
    ATE = torch.zeros((1,len(int_levels)))
    E_DO = torch.zeros((1,len(int_levels)))
    Training_time = torch.zeros((1,1))
    MMD = torch.zeros((1,len(int_levels)))
    KSD = torch.zeros((1,len(int_levels)))
    models = []
    model_base_dists = []

    # Drawing data
    torch.manual_seed(seed)
    noise_dists = [base_dist]*num_base_change + [Normal(0,1)]*(len(parents)-num_base_change)
    noise_dist_transforms = [base_transform]*num_base_change + [lambda x : x]*(len(parents)-num_base_change)
    Xobs,Xint = DGP(n, nint, True, intervention, int_levels, noise_dists,noise_dist_transforms)
    Xobs,Xobstest = Xobs[:n],Xobs[n:]

    
    # Estimating models
    start_time = time.time()
    for i in range(len(parents)):
            model_base_dist_i = copy.deepcopy(model_base_dist)
            # Getting nodes from graph
            index_x,index_y = parents[i],[i]
            inputs,outputs = Xobs[:,index_x].view(n,len(index_x)),Xobs[:,index_y].view(n,len(index_y))
    
            # Specifying models for cross-validation
            if use_RQS_base_flow:
                conditioners_validation = ([[NN(len(parents[i]),widths[m],layers),
                                            C(1,1,1),
                                            C(1,RQS_bins*3+2,3)] for m in range(len(widths))]+
                                           [[NN(len(parents[i]),widths[m],layers),
                                            NN(len(parents[i]),widths[m],layers),
                                            C(1,RQS_bins*3+2,3)] for m in range(len(widths))]+
                                            [[NN(len(parents[i]),widths[m],layers),
                                            NN(len(parents[i]),widths[m],layers),
                                            NN(len(parents[i]),widths[m],layers,RQS_bins*3+2)] for m in range(len(widths))])
                transformers_validation = [Transformer([Shift_layer(),
                                                        Scale_layer(),
                                                        RQS_layer(RQS_bins)],logdet = True)]*len(widths)*3
            else:
                conditioners_validation = ([[NN(len(parents[i]),widths[m],layers),
                                            C(1,1,1)] for m in range(len(widths))]+
                                            [[NN(len(parents[i]),widths[m],layers),
                                            NN(len(parents[i]),widths[m],layers)] for m in range(len(widths))]+
                                            [[NN(len(parents[i]),widths[m],layers),
                                            NN(len(parents[i]),widths[m],layers),
                                            NN(len(parents[i]),widths[m],layers, p = RQS_bins*3+2)] for m in range(len(widths))])
                transformers_validation = ([Transformer([Shift_layer(),Scale_layer()],logdet = True)]*len(widths)+
                                            [Transformer([Shift_layer(),Scale_layer()],logdet = True)]*len(widths)+
                                            [Transformer([Shift_layer(),Scale_layer(),RQS_layer(RQS_bins)],logdet = True)]*len(widths))
                
            models_validation = [flow_model(conditioners_validation[m],
                                       transformers_validation[m]) for m in range(len(widths))]
    
            loss_fn =  likelihood_loss(dist = model_base_dist_i, tail_adapt = tail_adapt)
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
            final_model[0].transformer.logdet = False
            models.append(final_model[0])
            if tail_adapt:
                df = model_base_dist_i.df[0].detach()
                model_base_dists.append(StudentT(df,0,1))
            else:
                model_base_dists.append(model_base_dist_i)
    Training_time[:,0] = time.time()-start_time
    
    # Interventional prediction
    for i in range(len(int_levels)):
        
        # Setting intervention level
        a = int_levels[i]
        
        # Getting interventional_samples
        Xpred,Xintpred = SCM_intervention_sample(parents,
                                                     models,
                                                     model_base_dists,
                                                     intervention,
                                                     [["id",a,"id","id"]],
                                                     mc_samples)
        # Getting ATEs                      
        E_DO[:,i] = Xintpred[0][:,-1].detach().mean() - Xint[i][:,-1].mean()
        ATE[:,i] = E_DO[:,i] - (Xpred[:,-1].detach().mean()-Xobstest[:,-1].mean())
        
        # Getting MMD
        Ypred,Y = Xintpred[0][:,-1:],Xint[i][:,-1:]
        MMD[:,i] = mmd(kernel = gaussian_kernel(1,1))(Y,Ypred,
            median_heuristic = True,
            mmd_samples = mmd_samples,
            heuristic_samples = heuristic_samples)                                 
        
        # Getting KSD
        KSD[:,i] = kolmogorov_distance(Ypred,Y)
        
    return { "names": model_name,
            "dataset": dataset,
            "base dist": base_dist,
            "EY|DO(X)": E_DO,
            "ATE": ATE,
            "MMD": MMD,
            "KSD": KSD,
            "training time": Training_time,
            "seed" : seed
           }
        

    