# Helper functions
import torch
from torch import nn
from torch.distributions import Normal,Uniform

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
        Xobs[:,i] = (models[i].transformation(Xobs[:,parents[i]].view(nsamples,len(parents[i])),
                                                  U[:,i].view(nsamples,1))).view(nsamples,).detach()
    
    # Getting interventional samples
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