import torch
from torch import nn
from torch.distributions import Normal,Uniform

class Empty_Conditioner:
    
    def __init__(self):
        self.params = torch.ones(1,requires_grad = True)
    
    def parameters(self):
        return [self.params]
    
    def forward(self,X):
        return 0
    
class Constant_Conditioner:
    
    def __init__(self,init = 0.0):
        self.params = torch.tensor([init],requires_grad = True)
    
    def parameters(self):
        return [self.params]
    
    def forward(self,X):
        return self.params

class Lin_Conditioner(nn.Module):
    def __init__(self,d,p,bias=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(d, p, bias)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.stack(x)
        return y

class RFF_Conditioner:
    
    def __init__(self,features,input_dims,sd=1):
        self.params = [Normal(0,sd/features).sample((features,1)).requires_grad_(True), # beta
                       Normal(0,sd/input_dims).sample((features,input_dims)).requires_grad_(True), # W
                       torch.ones((input_dims,),requires_grad = True)] # alpha
    
    def parameters(self):
        return self.params

    def forward(self, X : "N x D"):
        
        return (self.params[0].T @ torch.cos(self.params[1] @ (X*self.params[3]).T +self.params[2])).T
    
class NN_Conditioner(nn.Module):
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

pi = torch.acos(torch.zeros(1)).item() * 2 