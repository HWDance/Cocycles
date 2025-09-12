import torch
from torch.distributions import StudentT,Normal,Laplace, Bernoulli, Beta, Categorical, Uniform
from random import sample,seed
from scipy.special import hyperu
import pickle

from causal_cocycle.model import cocycle_model,flow_model
from causal_cocycle.optimise import *
from causal_cocycle.loss_functions import Loss
from causal_cocycle.conditioners import Constant_Conditioner, Lin_Conditioner,NN_RELU_Conditioner
from causal_cocycle.transformers import Transformer,Shift_layer,Scale_layer,RQS_layer
from causal_cocycle.helper_functions import likelihood_loss,mmd
from causal_cocycle.kernels import *

# Distribution functions:
class NBP:
    def __init__(self,alpha=1.0):
        self.alpha = alpha
        
    def sample(self, shape):
        lmbda = Beta(self.alpha,self.alpha).sample(shape)
        return Normal(0,lmbda/(1-lmbda)).sample()

class MixtureDist:

    def __init__(self,sd):
        self.sd = torch.tensor(sd,requires_grad = True)

    def sample(self, shape):
        B = Bernoulli(0.5).sample(shape)
        N = self.sd*Uniform(0.0,0.001).sample(shape)
        C = self.sd*Uniform(0.999,1.00).sample(shape)
        return B*N + (1-B)*C

class likelihood_loss:
    
    def __init__(self,dist):
        self.dist = dist
        if hasattr(dist,'df'):
            self.parameters = self.dist.df.requires_grad_(True)
        else:
            self.parameters = torch.tensor(1.0)
        self.loss_fn = "MLE"
    
    def __call__(self,model,inputs,outputs):
        U,logdet = model.inverse_transformation(inputs,outputs)
        
        return torch.mean(-self.dist.log_prob(U) - logdet) 

def main():
    n = 1000 # training samples
    d = 1 # input dims
    trials = 20 # experiment replications
    ngrid = 100 # grid points for CATE
    zlist = torch.linspace(-2,2,ngrid)[:,None]/d**0.5 # grid generation for CATE
    sd = 1.0 # \sigma^2 on Y|X,Z
    mc_samples = 10**5 # MC samples to approx expectations
    rng = 0
    
    # Shift interventions
    int_levels = torch.linspace(-2,2,50)
    
    # Training set up
    train_val_split = 1
    ntrain = int(n)
    learn_rate = [1e-3]
    scheduler = False
    val_tol = 1e-3
    batch_size = 128
    val_loss = False
    maxiter = 5000
    miniter = 5000
    RQS_bins = 8
    df_init = 10.0
    grad = False # learn conditioners
    
    # Model names
    Models = ["True", "G-RQS SCM","L-RQS SCM"]
    tail_adapt = [False,False,False,]
    
    # Fit Models
    KSDs = torch.zeros((len(Models)+1,len(int_levels),trials))
    Means = torch.zeros((len(Models)+1,len(int_levels),trials))
    for t in range(trials):
        
        # Seed set
        seed(t+rng)
        torch.manual_seed(t+rng)
        
        # Sample data
        U = MixtureDist(1.0).sample((n,1)).detach()

        # Models
        SCM_models = []
    
        # base distributions
        SCM_base_distributions = [
            MixtureDist(1.0),
            Normal(0,1),
            Laplace(0,1),
        ]
            
        # Transformers and conditioners
        SCM_conditioners = [
            [Constant_Conditioner(init = torch.log(torch.exp(torch.ones(1))-1), grad = grad, full = False)],
            [Constant_Conditioner(init = torch.ones(1), full = False),
              Constant_Conditioner(init = torch.ones(1), full = False),
              Constant_Conditioner(init = torch.ones((1,3*RQS_bins+2))*1.0),
              Constant_Conditioner(init = torch.ones(1), full = False),
              Constant_Conditioner(init = torch.ones(1), full = False)],
            [Constant_Conditioner(init = torch.ones(1), full = False),
              Constant_Conditioner(init = torch.ones(1), full = False),
              Constant_Conditioner(init = torch.ones((1,3*RQS_bins+2))*1.0),
              Constant_Conditioner(init = torch.ones(1), full = False),
              Constant_Conditioner(init = torch.ones(1), full = False)],
        ]
        SCM_transformers = [
            Transformer([Scale_layer()],logdet = True),
            Transformer([Shift_layer(),Scale_layer(),RQS_layer(),Shift_layer(),Scale_layer()],logdet = True),
            Transformer([Shift_layer(),Scale_layer(),RQS_layer(),Shift_layer(),Scale_layer()],logdet = True),
        ]
        
        
        # Model estimation
            
        # Getting input-output pairs
        inputs_train,outputs_train = U,U
        inputs_val,outputs_val = [],[]
    
        # SCM training
        for m in range(len(Models)):
            loss_fn = likelihood_loss(dist = SCM_base_distributions[m])
            model = flow_model(SCM_conditioners[m],SCM_transformers[m])
            if m > 0:
                model.transformer.logdet = True
                optimise(model,
                              loss_fn,
                              inputs_train,
                              outputs_train,
                              inputs_val,
                              outputs_val, 
                              batch_size = batch_size,
                              learn_rate = learn_rate,
                              print_ = False,
                              plot = False, 
                              miniter = miniter,
                              maxiter = maxiter, 
                              val_tol = val_tol,
                              val_loss = val_loss,
                              scheduler = scheduler,
                              likelihood_param_opt = True)
            model.transformer.logdet = False
            SCM_models.append(model)
    
            if tail_adapt[m]:
                SCM_base_distributions[m] = StudentT(SCM_base_distributions[m].df.detach(),0,1)
        
        print(t)
    
        # Interventional Distribution Construction + Evaluation
        for i in range(len(int_levels)):
    
            # Int sampling
            Y0pred = []
            for m in range(len(Models)):
                basepred = SCM_base_distributions[m].sample((mc_samples,1))
                Y0pred.append(torch.sigmoid(SCM_models[m].transformation(basepred,basepred).detach()+int_levels[i]))
                Means[m,i,t] = (Y0pred[0].mean()-Y0pred[m].mean()).abs()
            Y0pred.append(Y0pred[0][:n])
            Means[-1,i,t] = (Y0pred[0].mean()-Y0pred[-1].mean()).abs()
    
            # CDF Construction + Storage
            eps = torch.linspace(0,1,100)[None]
            cdfs = []
            for m in range(len(Models)+1):
                cdfs.append((Y0pred[m]<=eps).float().mean(0))
                KSDs[m,i,t] = (cdfs[0]-cdfs[m]).abs().max()

    return KSDs, Means
    
if __name__ == "__main__":
    KSDs, Means = main()
    print("KSD : ", KSDs)
    print("Mean : ", Means)
    data = {"KSD": KSDs, "Mean": Means}
    with open("results_supportmispec.pkl", "wb") as f: pickle.dump(data, f)