from torch.distributions import Normal,Categorical
import torch
import inspect

class CCDAGM:
    
    def __init__(self,models,parents):
        """
        models = tuple of d cocycle models
        parents = tuple of parent indices
        """
        self.models = models
        self.parents = parents
    
    def train_cocycle(self,X,loss,median_heuristic = False,RFF = False,n_RFF = 100,train_val_split = 0.8,
                      optimiser_args = [], optimiser_argvals = []):
        
        # Dimensions set up
        N = len(X)
        ntrain = int(train_val_split*len(X))
        
        for i in range(len(self.parents)):
            
            # Getting relevant variables from graph
            index_x,index_y = parents[i],[i]
            X,Y = Xobs[:,index_x].view(N,len(index_x)),Xobs[:,index_y].view(N,len(index_y))

            # Data Preprocessing
            inputs_train,outputs_train, inputs_val,outputs_val = X[:ntrain],Y[:ntrain],X[ntrain:],Y[ntrain:]

            # Setting up objective fn
            loss_fn = Loss(loss_fn = self.models[i],kernel = [gaussian_kernel(),
                                                                 gaussian_kernel()])
            if RFF_features:
                loss_fn.get_RFF_features()
            if median_heuristic:
                loss_fn.median_heuristic(inputs_train,outputs_train, subsamples = 10**4)
            
            # Amending optimisation settings
            args = torch.tensor(inspect.getfullargspec(Train.optimise)[0])
            arg_vals = list(inspect.getfullargspec(Train.optimise)[3])
            if optimiser_args:
                for j in range(len(optimiser_args)):
                    index = np.where(optimiser_args[j]==args)[0][0]-5
                    arg_vals[index] = optimiser_argvals[j]
            arg_vals = tuple([loss_fn,
                              inputs_train,
                              outputs_train,
                              inputs_val,
                              outputs_val]
                             +arg_vals)
            # Optimisation
            self.models[i] = Train(self.models[i]).optimise(*arg_vals)
    
    def conditional_expectation(self,Y_int,Z,regression_functional = [],
                                      optimiser_args = [], optimiser_argvals = [], train = True):
        
        # Initialising regressor
        CE_regressor = Conditional_Expectation_Regressor(regression_functional)
        
        # Updating optimiser arguments and training
        if train:
            args = torch.tensor(inspect.getfullargspec(CE_regressor.optimise)[0])
            arg_vals = list(inspect.getfullargspec(CE_regressor.optimise)[3])
            if optimiser_args:
                for j in range(len(optimiser_args)):
                    index = np.where(optimiser_args[j]==args)[0][0]-2
                    arg_vals[index] = optimiser_argvals[j]
            arg_vals = tuple([Z,Yint]+arg_vals)
            losses = CE_regressor.optimise(*arg_vals)
        
        return CE_regressor
        
    
    def distribution_transformation(self,X,intervention_function,intervention_level):
        """
        X : n x d tensor
        intervention_function : (phi,X) -> phi(X)
        intervention_level : phi
        
        Outputs h(phi,X)
        """
        k = len(self.models)
        n = len(X)
        X_int = X*1
        
        for i in range(k):
            pai = self.parents[i]
            if pai != []:
                npa = len(pai)
                Xi,Xpai,Xintpai = (X[:,i].view(n,1),
                                   X[:,pai].view(n,npa),
                                   X_int[:,pai].view(n,npa))
                X_int[:,i] = self.models[i].cocycle(Xintpai,Xpai,Xi).detach().view(n,)
            if intervention_level[i] != "id":
                X_int[:,i] = intervention_function(intervention_level[i],X_int[:,i])
        
        return X,X_int.detach()
    
    def interventional_dist_sample(self,X,intervention_function,intervention_level,samples,density = [],uniform_subsample = False):
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
        
    def counterfactual_dist_sample(self,X,intervention_function,intervention_level,samples,kernel = [], weights = []):
        """
        X : n x d tensor (subset of variables not conditioned on)
        intervention_function : (phi,X) -> phi(X)
        intervention_level : phi
        density : KDE or empirical distributon (leave empty for empirical)
        weights : vector of weights on local densities (P_i(X|z))_i=1^N
        samples : # MC samples
        """
        
        # Getting samples 
        Z = X[Categorical(weights).sample((samples,)),:]    
        Xsample = Z[:,~conditioning_set]
        if kernel != []:
            Xsample += Normal(0,1/kernel.lengthscale).sample((samples,))
        Xsample[:,conditioning_set] = conditioning_level
           
        # Passing through function
        return self.distribution_transformation(Xsample,intervention_function,intervention_level)
    
        
        
            
            
        
        