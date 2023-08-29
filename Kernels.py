import torch

class kernel:

    def __init__(self,lengthscale,scale):
        self.lengthscale = lengthscale
        self.scale = scale
        
    def get_gram(self, X : torch.tensor, Z : torch.tensor):
        # Computes K_XZ
        raise NotImplementedError

class gaussian_kernel(kernel):
        
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**2)            
        return K_xx*self.scale
    
class inverse_gaussian_kernel(kernel):
        
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X*self.lengthscale, Z*self.lengthscale, p=2.0)**2)            
        return K_xx*self.scale
    

class multivariate_gaussian_kernel(kernel):
    
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X @ self.lengthscale, Z @ self.lengthscale, p=2.0)**2)
        return K_xx*self.scale