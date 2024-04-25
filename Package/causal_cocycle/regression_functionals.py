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
    
    def __init__(self,kernel, penalty = 1e-2,reg = 1e-6):
        self.kernel = kernel
        self.reg = reg
        self.hyperparameters = [self.kernel.lengthscale,
                                torch.tensor([penalty], requires_grad = True)]
    
    def __call__(self,Ytrain,Xtrain,Xtest):
        ridge_penalty = self.hyperparameters[-1].abs()
        self.kernel.lengthscale = self.hyperparameters[0].abs()
        K_xx = self.kernel.get_gram(Xtrain,Xtrain)
        K_xtestx =  self.kernel.get_gram(Xtest,Xtrain)
        In = torch.eye(len(Ytrain))*(ridge_penalty+self.reg)
        
        return K_xtestx @ torch.linalg.solve(K_xx+In, Ytrain)        
    
class NW_functional(Functional):
    # Implements NW regression
    
    def __init__(self,kernel,reg = 0):
        self.kernel = kernel
        self.hyperparameters = [self.kernel.lengthscale]
        self.reg = reg
    
    def __call__(self, Ytrain,Xtrain,Xtest):
        
        # Setting parameters
        self.kernel.lengthscale = self.hyperparameters[0].abs()
        K_xtestx = self.kernel.get_gram(Xtest,Xtrain)+self.reg
        return K_xtestx @ Ytrain / K_xtestx.sum(1)[:,None] # to check

class LL_functional(Functional):
    # Implements local_lin regression
    
    def __init__(self,kernel,reg = 1e-6):
        self.kernel = kernel
        self.reg = reg
        self.hyperparameters = [self.kernel.lengthscale]
        
    def __call__(self, Ytrain,Xtrain,Xtest):
        self.kernel.lengthscale = torch.exp(self.hyperparameters[0])
        K_xtestx = self.kernel.get_gram(Xtest,Xtrain)
        Xtild = torch.column_stack((torch.ones((len(Xtrain),1)),Xtrain))
        Xtild_test = torch.column_stack((torch.ones((len(Xtest),1)),Xtest))
        
        ypred = torch.zeros(len(Xtest),1)
        for i in range(len(Xtest)):
            XWX = Xtild.T @ (Xtild * K_xtestx[i][:,None])
            ypred[i] = Xtild_test[i:i+1] @ torch.linalg.solve(XWX+torch.eye(2)*self.reg,Xtild.T @ (Ytrain * K_xtestx[i][:,None]))
            
        return ypred