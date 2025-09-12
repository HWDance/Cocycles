import torch
from torch.distributions import Normal,Uniform,OneHotCategorical,Gamma

class Mixture1D:
    
    def __init__(self,base_dists,probabilities,noints,scales):
        self.dists = base_dists
        self.probabilities = probabilities
        self.noints = noints
        self.scales = scales
        
    def sample(self,size):
        C = OneHotCategorical(self.probabilities).sample(size)[:,0]
        Z = torch.zeros((size[0],len(self.probabilities)))
        for i in range(len(self.dists)):
            Z[:,i] = self.noints[i]+self.scales[i]*self.dists[i].sample(size).T
        return (Z*C).sum(1)[:,None]          

def policy(V, flip_prob = 0.00):
    Z = (V.mean(1)*len(V.T)**0.5)[:,None]
    X_correct =  (Z<-1)*0+(Z>=-1)*(Z<1)*1 + (Z>=1)*2
    flips = (Uniform(0,1).sample((len(V),1))<flip_prob)*1
    return X_correct*(1-flips) + torch.randint(3, (len(V),1))*flips

def new_policy(V, flip_prob = 0.00):
    Z = (V.mean(1)*len(V.T)**0.5)[:,None]
    X_correct =  (Z<-1)*0+(Z>=-1)*1
    flips = (Uniform(0,1).sample((len(V),1))<flip_prob)*1
    return X_correct*(1-flips) + torch.randint(2, (len(V),1))*flips

def shift(V,policy,projection_coeffs):
    t = policy(V)
    z = V @ projection_coeffs
    return 1/(1+torch.exp(z)) + ((t==0)*torch.exp(-0.1*(z+3)**2) + 
                                 (t==1)*torch.exp(-0.1*(z-0)**2)*0.75 + 
                                 (t==2)*torch.exp(-0.1*(z-3)**2)*0.5)

def scale(V,projection_coeffs):
    z = V @ projection_coeffs
    return 0.1*(torch.exp(-1/10*(z+2)**2*(z-2)**2)+1)

def DGP(N,D,policy,projection_coeffs,covariate_corr = 0, 
        covariate_dist = Normal(0,1),
        noise_dist = Normal(0,1)):
    Sigma = (1-covariate_corr)*torch.eye(D)+covariate_corr*torch.ones((D,D))
    A = torch.linalg.cholesky(Sigma)
    Z = covariate_dist.sample((N,D)) @ A.T
    U = noise_dist.sample((N,1))
    Y = shift(Z,policy,projection_coeffs) + scale(Z,projection_coeffs)*U
    X = torch.column_stack((policy(Z),Z))
    return Z,X,Y