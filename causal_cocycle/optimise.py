# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:34:25 2023

@author: hughw
"""
import torch
import numpy as np
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import os
import inspect
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
def get_subsample(inputs,outputs=None,subsamples=None):
        ind_list = np.linspace(0,len(inputs)-1,len(inputs)).astype(int)
        batch_inds = torch.tensor([np.random.choice(ind_list,subsamples)]).long().view(subsamples,)
        inputs_batch = inputs[batch_inds]
        if outputs!=None:
            outputs_batch = outputs[batch_inds]
            return inputs_batch,outputs_batch
        else:
            return inputs_batch

"""
General optimisation function for coboundary models/ CTMs / BCMs with flows
"""
def optimise(model,
             loss,
             inputs,
             outputs, 
             loss_val = [],
             inputs_val = [],
             outputs_val = [],
             learn_rate = [1e-3], 
             maxiter = 10000,
             miniter = 10000, 
             weight_decay = 0,
            # optimise_loss_params = True, 
             val_loss = True, 
             val_loss_freq = 100,
             val_tol = 1e-3, 
             batch_size = 1024, 
             val_batch_size = 1024,
             scheduler = False, 
             schedule_milestone = 100, 
             n_schedule = 100,
             lr_mult = 0.90, 
             plot = False, 
             plot_start = 30, 
             print_ = False, 
             optimise_conditioners = True, 
             likelihood_param_opt = False,
             likelihood_param_lr = 0.01):

    # Dimensions
    m = len(outputs)
    m_val = len(outputs_val)

    # Parameters set up
    conditioner_params_list = []
    if len(learn_rate)==1:
        learn_rate = learn_rate*len(model.conditioner)
    for k in range(len(model.conditioner)):
            if optimise_conditioners or (len(optimise_conditioners)>1 and optimise_conditioners[k]==True):
                    conditioner_params_list +=  [{'params' : model.conditioner[k].parameters(),
                                                                        'lr' : learn_rate[k]}]  
    if likelihood_param_opt:
         conditioner_params_list +=  [{'params' : [loss.parameters],
                                        'lr' : likelihood_param_lr}]  
    # Optimiser set up
    optimizer = torch.optim.Adam(conditioner_params_list,weight_decay = weight_decay)  
    if scheduler:
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                      step_size = schedule_milestone, 
                                                      gamma=lr_mult)
        
    # Optimisation iterations
    Losses = torch.zeros(maxiter)
    Losses_val = torch.zeros(maxiter)
    i = 0 # iter counter
    
    while i < maxiter:
        optimizer.zero_grad()

        #Subsampling
        if batch_size < m:
            inputs_batch,outputs_batch = get_subsample(inputs,
                                                       outputs,
                                                       batch_size)

        # Training loss computation
        try:
            Loss = loss(model,inputs_batch,outputs_batch)
        except:
            print("Forward pass error: exiting training iterations")
            break    
        Losses[i] = Loss.detach()

        # Optimisation step
        Loss.backward()
        optimizer.step()
        if scheduler:
            lr_schedule.step()

        # Display
        if print_ or plot:
            if not i % 10:
                clear_output(wait=True)
                if print_:
                    print("Training loss last 10 avg is :",Losses[i-10:i].mean())
                    print(100*i/maxiter," % completion")
                if plot and i > plot_start:
                    plt.plot(Losses[20:i+1])
                    display(plt.gcf())
        i += 1

    # Validation loss computation
    objs = []
    if val_loss and inputs_val != []:
        if loss_val == []:
            loss_val = loss
        with torch.no_grad():
            objs.append(loss_val(model,inputs_val[:val_batch_size],outputs_val[:val_batch_size]))
        
    return objs
    
def get_CV_splits(X,folds = 5):
    """
    X: nxd matrix
    folds: # CV folds 

    Returns: list of folds x training and validation sets
    """

    n = len(X)
    n_per_fold = int(n/folds+1-1e-10) # rounds up except if exact integer
    row_count = torch.linspace(0,n-1,n) 
    train_val_sets = list()

    for i in range(folds):
        test_inds = ((row_count>= n_per_fold*i)*(row_count<n_per_fold*(i+1)))>0
        train_inds = test_inds==0
        train_val_sets.append([X[train_inds],X[test_inds]])

    return train_val_sets

"""
For cross-validation over models
"""
def validate(models,loss,inputs,outputs,loss_val =[],method = "CV", train_val_split = 0.8,opt_args=[],opt_argvals=[],hyper_args=[],hyper_argvals=[],
            choose_best_model = "overall", retrain = True):
    """
    models : list of cocycle models 
    loss : a Loss object for training
    inputs : N x D tensor of inputs X
    outputs : N x P tensor of outputs Y
    loss_val : a Loss object for validation
    method : "CV" for k-fold CV, "fixed" for a fixed validation set
                with size determined by (1-train_val_split) x n
    train_val_split : split for K fold CV or train/validation set
    opt_args : list of arguments for optimise() to change from defaults
    opt_argvals: list of values for opt_args
    hyper_args : list of hyperparameters for optimise() to change from defaults per each model
    hyper_argvals: list of list of hyperparameter values per each model
    choose_best_model : str ("overall" = choose best hypers across all folds, 
                             "per fold" = choose best hypers per fold)
    retrain : Bool (True = retrain best model on final dataset,
                            only runs for choose_best_model = "overall")
    """ 
    
    # Checking dims
    n,d = inputs.size()
    N,p = outputs.size()
    assert(n==N)
    if hyper_args:
        assert(len(models) == len(hyper_args))
        assert(len(hyper_args) == len(hyper_argvals))
    assert(len(opt_args) == len(opt_argvals))
    
    # Getting CV folds optionally
    if method == "CV":
        folds = int(1/(1-train_val_split))
        input_splits = get_CV_splits(inputs,folds)
        output_splits = get_CV_splits(outputs,folds)
    else:
        folds = 1
        ntrain = int(train_val_split*n)
        input_splits = [[inputs[:ntrain],inputs[ntrain:]]]
        output_splits = [[outputs[:ntrain],outputs[ntrain:]]]
    
    # Setting optimisation arguments
    optimiser_args = inspect.getfullargspec(optimise)[0]
    optimiser_defaults = inspect.getfullargspec(optimise)[3]
    num_non_defaults = len(optimiser_args) - len(optimiser_defaults)
    new_argvals = list(optimiser_defaults)
    if opt_args:
        for j in range(len(opt_args)):
            index = (np.where(opt_args[j]==np.array(optimiser_args))[0][0]
                    -num_non_defaults)
            new_argvals[index] = opt_argvals[j]

    # Optimisation
    Val_losses = torch.zeros((len(models),folds))
    Models_store = []
    for m in range(len(models)):

        models_store = []
        
        # Updating hyperparameters for optimiser for model m
        if hyper_args:
            hyper_arg,hyper_argval = hyper_args[m],hyper_argvals[m]
            if hyper_arg:
                for j in range(len(hyper_arg)):
                    hyper_index =  (np.where(hyper_arg[j]==np.array(optimiser_args))[0][0]
                                    -num_non_defaults)
                    new_argvals[hyper_index] = hyper_argval[j]
        
        # Doing CV
        for k in range(folds):
            
            # Getting latest initialised model
            model = copy.deepcopy(models[m])
            
            # Getting train/val split for kth fold
            inputs_train,outputs_train = input_splits[k][0],output_splits[k][0]
            inputs_val,outputs_val = input_splits[k][1],output_splits[k][1]
            
            # Finalising optimiser arguments by adding non-defaults 
            new_argvals[0],new_argvals[1],new_argvals[2] = loss_val,inputs_val,outputs_val
            final_args = [model,loss,inputs_train,outputs_train]+new_argvals
                
            # Optimising model
            Val_losses[m,k] = optimise(*final_args)[0]
            print("Currently optimising model ",m,", for fold ",k)
        
            # Storing model
            models_store.append(model)

        Models_store.append(models_store)
    
    # Getting best models
    Val_losses[torch.isnan(Val_losses)] = float('inf') 
    if choose_best_model == "overall":
        best_ind = torch.where(Val_losses.mean(1) ==Val_losses.mean(1).min())[0][0]
        final_model = Models_store[best_ind]
    else:
        final_model = []
        for k in range(folds):
            best_ind_k = torch.where(Val_losses[:,k] ==Val_losses[:,k].min())[0][0]
            final_model.append(Models_store[best_ind_k][k])

    # Optional retraining on full dataset when selecting overall model
    if retrain and choose_best_model == "overall":
        
        # Setting correct hypers
        if hyper_args:
            hyper_arg,hyper_argval = hyper_args[best_ind],hyper_argvals[best_ind]
            if hyper_arg:
                for j in range(len(hyper_arg)):
                    hyper_index =  (np.where(hyper_arg[j]==np.array(optimiser_args))[0][0]
                                    -num_non_defaults)
                    new_argvals[hyper_index] = hyper_argval[j]

        # Getting initialised model
        model = copy.deepcopy(models[best_ind])
        
        # Finalising optimiser arguments by adding non-defaults, and using full training set
        new_argvals[0],new_argvals[1],new_argvals[2] = loss_val,[],[]
        final_args = [model,loss,inputs,outputs]+new_argvals
            
        # Optimising final model
        optimise(*final_args)
        print("Finished optimising final model")
    
        # Storing final model and avg val loss
        final_model = model
        Val_losses = Val_losses.nanmean(1).min()

    return final_model,Val_losses
        
    
    
    
    
                

    