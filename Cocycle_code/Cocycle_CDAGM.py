from torch.distributions import Normal,Categorical
import torch

class CCDAGM:
    
    def __init__(self,cocycle_models,parents):
        """
        cocycles = tuple of d cocycle models
        parents = tuple of parent indices
        """
        self.cocycle_models = cocycle_models
        self.parents = parents
    
    def distribution_transformation(self,X,intervention_function,intervention_level):
        """
        X : n x d tensor
        intervention_function : (phi,X) -> phi(X)
        intervention_level : phi
        
        Outputs h(phi,X)
        """
        k = len(self.cocycle_models)
        n = len(X)
        X_int = X*1
        
        for i in range(k):
            pai = self.parents[i]
            if pai != []:
                npa = len(pai)
                Xi,Xpai,Xintpai = (X[:,i].view(n,1),
                                   X[:,pai].view(n,npa),
                                   X_int[:,pai].view(n,npa))
                X_int[:,i] = self.cocycle_models[i].cocycle(Xintpai,Xpai,Xi).detach().view(n,)
            if intervention_level[i] != "id":
                X_int[:,i] = intervention_function(intervention_level[i],X_int[:,i])
        
        return X,X_int.detach()
    
    def interventional_dist_sample(self,X,intervention_function,intervention_level,samples,density = [],uniform_subsample = True):
        """
        X : n x d tensor
        intervention_function : (phi,X) -> phi(X)
        intervention_level : phi
        samples : # MC samples
        density : KDE or empirical distributon (leave empty for empirical)
        uniform_subsample : True = uniformly subsample data points
        """
    
        # Getting samples
        if uniform_subsample: 
            Z = X[torch.randint(0,len(X),(samples,))]
        else:
            Z = X
            
        if density == []:
            Xsample = Z
        else:
            Xsample = Z+Normal(0,1/density.kernel.lengthscale.abs()).sample((samples,))
           
        # Passing through function
        return self.distribution_transformation(Xsample,intervention_function,intervention_level)
    
    def counterfactual_dist_sample(self,X,intervention_function,intervention_level,conditioning_set,conditioning_level,samples,density = [], uniform_subsample = True):
        """
        X : n x d tensor (subset of variables not conditioned on)
        intervention_function : (phi,X) -> phi(X)
        intervention_level : phi
        conditioning_set : list of variables to be conditioned on
        conditioning_levels : list of values to condition on
        density : KDE or empirical distributon (leave empty for empirical)
        samples : # MC samples
        uniform_subsample : True = uniformly subsample data points (always true for conditional KDE)
        """
    
        # Getting samples
        if density == []:
            valid_empirical_samples = (X[:,conditioning_set] == conditioning_level)
            Xvalid = X[valid_empirical_samples] 
            if uniform_subsample:
                Xsample = Xvalid[torch.randint(0,len(Xvalid),(samples,))]
            else:
                Xsample = Xvalid
        else:
            Z = X[Categorical(density.weights).sample((samples,)),:]    
            Xsample = Z[:,~conditioning_set]+Normal(0,1/density.kernel.lengthscale).sample((N,))
            Xsample[:,conditioning_set] = conditioning_level
           
        # Passing through function
        return self.distribution_transformation(Xsample,intervention_function,intervention_level)
    
        
        
            
            
        
        