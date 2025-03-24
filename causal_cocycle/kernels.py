import torch

class kernel:

    def __init__(self,lengthscale=[],scale=[]):
        if lengthscale == []:
            self.lengthscale = torch.ones(1,requires_grad = True)
        else:
            self.lengthscale = lengthscale
        if scale ==[]:
            self.scale = torch.ones(1)
        else:
            self.scale = scale
        
    def get_gram(self, X : torch.tensor, Z : torch.tensor):
        # Computes K_XZ
        raise NotImplementedError

class gaussian_kernel(kernel):
        
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**2)            
        return K_xx*self.scale
    
class exponential_kernel(kernel):
        
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X/self.lengthscale, Z/self.lengthscale, p=2.0)**1)            
        return K_xx*self.scale
    
class inverse_gaussian_kernel(kernel):
        
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X*self.lengthscale, Z*self.lengthscale, p=2.0)**2)            
        return K_xx*self.scale
    

class multivariate_gaussian_kernel(kernel):
    
    def get_gram(self,X,Z):
        K_xx = torch.exp(-0.5*torch.cdist(X @ self.lengthscale, Z @ self.lengthscale, p=2.0)**2)
        return K_xx*self.scale

class linear_kernel(kernel):

    def get_gram(self,X,Z):
        return X @ Z.T
    
def median_heuristic(X):
    """
    Returns median heuristic lengthscale for Gaussian kernel
    """

    # Median heurstic for inputs
    Dist = torch.cdist(X,X, p = 2.0)**2
    Lower_tri = torch.tril(Dist, diagonal=-1)
    mask = torch.tril(torch.ones_like(Lower_tri, dtype=torch.bool), -1)
    return (Lower_tri[mask].median()/2).sqrt()