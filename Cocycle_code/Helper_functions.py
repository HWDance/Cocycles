# Helper functions
import torch
from torch import nn
from torch.distributions import Normal,Uniform


class mmd:
    
    def __init__(self,kernel):
        self.kernel = kernel
    
    def get_median_heuristic(self,inputs,subsamples = 10000):
        """
        Returns median heuristic lengthscale for Gaussian kernel
        """
        
        inputs_batch = inputs[:subsamples]
        
        # Median heurstic for inputs
        Dist = torch.cdist(inputs_batch,inputs_batch, p = 2.0)**2
        Lower_tri = torch.tril(Dist, diagonal=-1).view(len(inputs_batch)**2).sort(descending = True)[0]
        Lower_tri = Lower_tri[Lower_tri!=0]
        self.kernel.lengthscale = (Lower_tri.median()/2).sqrt()
            
        return
        
    def __call__(self,X,Y,median_heuristic = True,mmd_samples = 5000,heuristic_samples = 1000):
        if median_heuristic:
            self.get_median_heuristic(X,heuristic_samples)
        return (self.kernel.get_gram(X[:mmd_samples],X[:mmd_samples]).mean() 
                + self.kernel.get_gram(Y[:mmd_samples],Y[:mmd_samples]).mean()
                -2*self.kernel.get_gram(X[:mmd_samples],Y[:mmd_samples]).mean())**0.5

class likelihood_loss:
    
    def __init__(self,dist, tail_adapt = False,tail_init = 1.0,log_det = True):
        self.dist = dist
        self.tail_adapt = tail_adapt
        self.parameters = torch.tensor([tail_init]).requires_grad_(True)
        self.log_det = log_det
    
    def __call__(self,model,inputs,outputs):
        if self.log_det:
            U,logdet = model.inverse_transformation(inputs,outputs)
            if self.tail_adapt:
                self.dist.df = torch.exp(self.parameters)
            return torch.mean(-self.dist.log_prob(U) - logdet)
        else:
            U = model.inverse_transformation(inputs,outputs)
            if self.tail_adapt:
                self.dist.df = torch.exp(self.parameters)
            return torch.mean(-self.dist.log_prob(U))

def SCM_intervention_sample(parents,models,base_distributions,intervention,intervention_levels,nsamples,interventional_sample = True):
    """
    parents, models, base distributions : list of appropriate objects (model is a cocycle model)
    intervention : function (a,x) -> f_a(x)
    intervention_levels : l x d list, l levels, d variables
    nsamples : # MC samples to draw
    interventional_sample : True = draw interventional samples
    """
    
    # Getting base samples
    U = torch.zeros((nsamples,len(parents)))
    for i in range(len(parents)):
        U[:,i] = base_distributions[i].sample((nsamples,))
        
    # Geting observational samples
    Xobs = torch.zeros((nsamples,len(parents)))
    for i in range(len(parents)):
        Xobs[:,i] = (models[i].transformation(Xobs[:,parents[i]].view(nsamples,len(parents[i])),
                                                  U[:,i].view(nsamples,1))).view(nsamples,).detach()
    # Getting interventional samples
    if interventional_sample:
        Xint = []
        for a in range(len(intervention_levels)):
            xint = torch.zeros((nsamples,len(parents)))
            for i in range(len(parents)):
                xint[:,i] = (models[i].transformation(xint[:,parents[i]].view(nsamples,len(parents[i])),
                                                      U[:,i].view(nsamples,1))).view(nsamples,).detach()
                if intervention_levels[a][i] != "id":
                    xint[:,i] = intervention(intervention_levels[a][i],xint[:,i])
            Xint.append(xint)
    
        return Xobs,Xint
    
    else:
        return Xobs
    
class propensity_score:
    
    def __init__(self,P,policy):
        self.P = P
        self.policy = policy
    """
    Pij = P(X = i |X* = j) is conditional dist on assignments `post error'
    given initial policy assignment
    """
    def __call__(self,X,V):
        assert(len(X) == len(V))
        col_select = self.policy(V,flip_prob = 0)[:,0]
        row_select = torch.linspace(0,len(X)-1,len(X)).int()
        conditional_dists = (self.P[...,None] @ torch.ones((1,1,len(V))))[:,col_select,row_select]
        return conditional_dists[X[:,0].int(),row_select]
    

class outcome_model:
    
    """ 
    Takes in a cocycle/flow model and defines an outcome model
    """
    
    def __init__(self,model,U_hat):
        self.model = model
        self.U_hat = U_hat # sampled noise
        
    def __call__(self,inputs, statistic):
        """
        Statistic \theta : (m x n) -> (d x m x n) is transformation \theta(Y) of interest
        """
        prediction = self.model.transformation_outer(inputs,self.U_hat)
        return statistic(prediction).mean(-1)
        