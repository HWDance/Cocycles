# Imports
import torch
import numpy as np
import random

# Causal_cocycle imports
from causal_cocycle.model_factory import FlowFactory
from causal_cocycle.loss import FlowLoss
from causal_cocycle.optimise_new import validate, optimise
from causal_cocycle.kernels import gaussian_kernel
from causal_cocycle.helper_functions import kolmogorov_distance

# Import the DGP and policy functions from BD.py
from BD import DGP, policy, new_policy

# Import configs 
from BD_scm_config_ta import model_config, opt_config
from BD_dgp_config import dgp_config

def run(seed, n = None, d = None, base = None):

    # Updating hypers
    if n is not None:
        dgp_config["N"] = n

    if d is not None:
        dgp_config["D"] = d
    
    if base == "Normal":
        dgp_config["noise_dist"] = torch.distributions.Normal(0,1)
    
    # Generating observational data with configs
    N = int(dgp_config['N']/2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    Z,X,Y = DGP(policy = policy,**dgp_config)
    Z1,X1,Y1 = Z[:N], X[:N],Y[:N]
    Z2,X2,Y2 = Z[N:], X[N:],Y[N:]
    
    # Generating interventional data with configs
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    Zint,Xint,Yint = DGP(policy = new_policy,**dgp_config)
    Zint1,Xint1,Yint1 = Zint[:N], Xint[:N],Yint[:N]
    Zint2,Xint2,Yint2 = Zint[N:], Xint[N:],Yint[N:]
    
    # Random dataset shuffle
    id = torch.randperm(Z.size(0))
    Ztr,Xtr,Ytr = Z[id],X[id],Y[id]
    
    # Model construction
    input_dim = dgp_config['D']
    factory = FlowFactory(input_dim+1, model_config)
    models, hyper_args = factory.build_models()
    print(f"Constructed {len(models)} candidate models.")
    
    # Loss construction
    loss= FlowLoss()
    
    # 1. Best overall model (choose one candidate based on average CV loss, then retrain on full training set)
    final_model_overall, (best_index_overall, best_loss_overall) = validate(
        models, 
        loss, 
        Xtr, Ytr,
        loss_val=loss,
        method="fixed",
        train_val_split=0.8,  # 2-fold CV: 50% training per fold.
        opt_kwargs=opt_config,
        hyper_kwargs=hyper_args,
        choose_best_model="overall",
        retrain=True
    )
    
    # 2. Best model per fold (select one candidate per fold without retraining)
    final_models_per_fold, (best_index_per_fold, best_loss_per_fold) = validate(
        models, 
        loss, 
        Xtr, Ytr,
        loss_val=loss,
        method="CV",
        train_val_split=0.5,
        opt_kwargs=opt_config,
        hyper_kwargs=hyper_args,
        choose_best_model="per fold",
        retrain=False
    )
    
    # Predicting counterfactuals+differences
    for f in final_models_per_fold:
        f.transformer.logdet = False
    final_model_overall.transformer.logdet = False
        
    # Predicting counterfactuals+differences
    def feature(y):
        return torch.log(1+torch.exp(-y))
    
    U1hat = final_models_per_fold[1].base_distribution.sample((N,1))
    U2hat = final_models_per_fold[0].base_distribution.sample((N,1))
    Uhat = final_model_overall.base_distribution.sample((2*N,1))    
    Yint1pred = final_models_per_fold[1].transformation(Xint1, U1hat).detach()
    Yint2pred = final_models_per_fold[0].transformation(Xint2, U2hat).detach()
    Yintpred_split = torch.concatenate((Yint1pred,Yint2pred))
    Yintpred = final_model_overall.transformation(Xint,Uhat).detach()
    
    Ycf1pred = final_models_per_fold[1].cocycle(Xint1,X1,Y1).detach()
    Ycf2pred = final_models_per_fold[0].cocycle(Xint2,X2,Y2).detach()
    Ycfpred_split = torch.concatenate((Ycf1pred,Ycf2pred))
    Ycfpred = final_model_overall.cocycle(Xint,X,Y).detach()
    
    counterfactual_diffs = feature(Yint[X[:,0]==2])-feature(Y[X[:,0]==2])
    counterfactual_diffpreds = feature(Ycfpred[X[:,0]==2])-feature(Y[X[:,0]==2])
    counterfactual_diffpreds_split = feature(Ycfpred_split[X[:,0]==2])-feature(Y[X[:,0]==2])
    
    # Constructing KSD
    KSD = kolmogorov_distance(counterfactual_diffs,counterfactual_diffpreds)
    KSDsplit = kolmogorov_distance(counterfactual_diffs,counterfactual_diffpreds_split)
    KSDint = kolmogorov_distance(Yintpred,Yint)
    KSDintsplit = kolmogorov_distance(Yintpred_split,Yint)
    
    obj = {
        "name":"flows_tail_adaptive",
        "seed": seed,
        "model_index": best_index_overall,
        "model_index_split": best_index_per_fold,
        "counterfactual_diffs": counterfactual_diffs,
        "counterfactual_diffpreds": counterfactual_diffpreds,
        "counterfactual_diffpreds_split": counterfactual_diffpreds_split,
        "KSD": KSD,
        "KSDsplit": KSDsplit,
        "KSDint": KSDint,
        "KSDintsplit": KSDintsplit
    }

    return obj

if __name__ == "__main__":
    run(seed = 0)