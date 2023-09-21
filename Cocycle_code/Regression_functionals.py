import torch

class Functional:
    
    def __init__(self,kernel):
        self.kernel = kernel
        raise NotImplementedError
    def __call__(self,Ytrain : torch.tensor,Xtrain : torch.tensor,Xtest : torch.tensor):
        # x -> f(x)
        raise NotImplementedError
        
    def get_weights(self,Ytrain : torch.tensor,Xtrain : torch.tensor):
        # Data  -> weights(Data)
        raise NotImplementedError
        
    def get_features(self,Xtrain : torch.tensor):
        # Data -> Phi(.)
        raise NotImplementedError
        
        
class KRR_functional(Functional):
    # Implements KRR
    
    def __init__(self,kernel, penalty = 1e-2):
        self.kernel = kernel
        self.hyperparameters = [self.kernel.lengthscale,
                                torch.tensor([penalty], requires_grad = True)]
    
    def __call__(self, Ytrain,Xtrain,Xtest):
        ridge_penalty = self.hyperparameters[-1]**2
        K_xx = self.kernel.get_gram(Xtrain,Xtrain)
        K_xtestx =  self.kernel.get_gram(Xtest,Xtrain)
        In = torch.eye(len(Ytrain))*ridge_penalty
        
        return K_xtestx @ torch.linalg.solve(K_xx+In, Ytrain)
    
    def get_weights(self,Ytrain,Xtrain):
        ridge_penalty = self.hyperparameters[-1]**2
        K_xx = self.kernel.get_gram(Xtrain,Xtrain)
        In = torch.eye(len(Ytrain))*ridge_penalty
        
        return torch.linalg.solve(K_xx+In, Ytrain)
    
    def get_features(self,Xtrain):
        self.Xtrain = Xtrain
        
        def feature(self,Xtest):
            return self.kernel.get_gram(Xtest,self.Xtrain)
        
        return feature
        
    
class NW_functional(Functional):
    # Implements NW regression
    
    def __init__(self,kernel):
        self.kernel = kernel
        self.hyperparameters = [self.kernel.lengthscale]
    
    def __call__(self, Ytrain,Xtrain,Xtest):
        
        # Setting parameters
        self.kernel.lengthscale = self.hyperparameters[0]**2
        K_xtestx = self.kernel.get_gram(Xtest,Xtrain)
        return K_xtestx @ Ytrain / K_xtestx.sum(1)[:,None] # to check
    
    def get_weights(self,Ytrain,Xtrain):
        return Ytrain
            
    def get_features(self,Xtrain):
        self.Xtrain = Xtrain
        
        def feature(self,Xtest):
            K_xtestx = self.kernel.get_gram(Xtest,self.Xtrain)
            return K_xtextx/K_xtextx.sum(1)[:,None]
        
        return feature   

class LL_functional(Functional):
    # Implements local_lin regression
    
    def __init__(self,kernel):
        self.kernel = kernel
        self.hyperparameters = [self.kernel.lengthscale]
        
    def __call__(self, Ytrain,Xtrain,Xtest):
        K_xtestx = self.kernel.get_gram(Xtest,Xtrain)
        Xtild = torch.column_stack((torch.ones((len(Xtrain),1)),Xtrain))
        Xtild_test = torch.column_stack((torch.ones((len(Xtest),1)),Xtest))
        
        ypred = torch.zeros(len(Xtest),1)
        for i in range(len(Xtest)):
            XWX = Xtild.T @ (Xtild * K_xtestx[i][:,None])
            ypred[i] = Xtild_test[i:i+1] @ torch.linalg.solve(XWX,Xtild.T @ (Ytrain * K_xtestx[i][:,None]))
            
        return ypred

    def get_weights(self,Ytrain,Xtrain):
        return Ytrain
            
    def get_features(self,Xtrain):
        self.Xtrain = Xtrain
        
        def feature(self,Xtest):
            K_xtestx = self.kernel.get_gram(Xtest,self.Xtrain)
            Xtild = torch.column_stack((torch.ones((len(Xtrain),1)),Xtrain))
        
            weights = torch.zeros(len(Xtest),1)
            for i in range(len(Xtest)):
                XWX = Xtild.T @ (Xtild * K_xtestx[i][:,None])
                weights[i] = torch.linalg.solve(XWX,(Xtild * K_xtestx[i][:,None]).T)

            return weights   
        return feature        