# Helper functions
import torch
from torch import nn
from torch.distributions import Normal,Uniform

class Lin(nn.Module):
    def __init__(self,neurons,d):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(d, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.stack(x)
        return y
    
class RFF:
    
    def __init__(self,features,input_dims,sd=1):
        self.params = [Normal(0,sd/features).sample((features,1)).requires_grad_(True), # beta
                       Normal(0,sd/input_dims).sample((features,input_dims)).requires_grad_(True), # W
                       Uniform(0,2*pi).sample((features,1)).requires_grad_(True), # b
                       torch.ones((input_dims,),requires_grad = True)] # alpha
    
    def parameters(self):
        return self.params

    def forward(self, X : "N x D"):
        
        return (self.params[0].T @ torch.cos(self.params[1] @ (X*self.params[3]).T +self.params[2])).T
    
class NN_sig(nn.Module):
    def __init__(self,nin,nh,nout):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(nin, nh),
            nn.Sigmoid(),
            nn.Linear(nh, nout),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        y = self.stack(x)
        return y
    
class NN_relu(nn.Module):
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            *([nn.Linear(input_dims, width)]+
              [nn.ReLU(),nn.Linear(width, width)]*(layers-1)+
              [nn.ReLU(), nn.Linear(width, output_dims, bias = bias)]),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.stack(x)
        return y
    
    
class NN_elu(nn.Module):
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            *([nn.Linear(input_dims, width)]+
              [nn.ELU(),nn.Linear(width, width)]*(layers-1)+
              [nn.ELU(), nn.Linear(width, output_dims, bias = bias)]),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.stack(x)
        return y
    
class NN_arctan(nn.Module):
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            *([nn.Linear(input_dims, width)]+
              [nn.Tanh(),nn.Linear(width, width)]*(layers-1)+
              [nn.Tanh(), nn.Linear(width, output_dims, bias = bias)]),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.stack(x)
        return y
    
pi = torch.acos(torch.zeros(1)).item() * 2 

def SCM_intervention_sample(parents,models,base_distributions,intervention,intervention_levels,nsamples):
    """
    parents, models, base distributions : list of appropriate objects (model is a cocycle model)
    intervention : function (a,x) -> f_a(x)
    intervention_levels : l x d list, l levels, d variables
    nsamples : # MC samples to draw
    """
    
    # Getting base samples
    U = torch.zeros((nsamples,len(parents)))
    for i in range(len(parents)):
        U[:,i] = base_distributions[i].sample((nsamples,))
        
    # Geting observational samples
    Xobs = torch.zeros((nsamples,len(parents)))
    for i in range(len(parents)):
        if parents[i]!= []:
            Xobs[:,i] = (models[i].transformation(Xobs[:,parents[i]].view(nsamples,len(parents[i])),
                                                  U[:,i].view(nsamples,1))).view(nsamples,).detach()
        else:
            Xobs[:,i] = U[:,i]  
    
    # Getting interventional samples
    Xint = []
    for a in range(len(intervention_levels)):
        xint = torch.zeros((nsamples,len(parents)))
        for i in range(len(parents)):
            if parents[i]!= []:
                xint[:,i] = (models[i].transformation(xint[:,parents[i]].view(nsamples,len(parents[i])),
                                                  U[:,i].view(nsamples,1))).view(nsamples,).detach()
            else:
                xint[:,i] = U[:,i] 
            if intervention_levels[a][i] != "id":
                xint[:,i] = intervention(intervention_levels[a][i],xint[:,i])
        Xint.append(xint)
    
    return Xobs,Xint

def median_heuristic(self,X):
    """
    Returns median heuristic lengthscale for Gaussian kernel
    """

    # Median heurstic for inputs
    Dist = torch.cdist(X,X, p = 2.0)**2
    Lower_tri = torch.tril(Dist, diagonal=-1).view(len(X)**2).sort(descending = True)[0]
    Lower_tri = Lower_tri[Lower_tri!=0]
    return (Lower_tri.median()/2).sqrt()