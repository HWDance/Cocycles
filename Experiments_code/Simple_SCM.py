import torch 
from torch.distributions import Normal,Laplace,LogNormal,Beta,StudentT,Uniform,Gamma

def simple_SCM(N = 1000, Nint = 10**6,interventional_data = False, intervention_function = [], levels = [], adversarial = False, scale = False, R2=0.5):
    
    m = N + Nint
               
    # Getting noise
    Z0 = Normal(0,1).sample((m,))
    if adversarial:
        Z1 = torch.sigmoid(StudentT(2,5,0.5).sample((m,)))
        Z2 = torch.sigmoid(StudentT(2,5,0.5).sample((m,)))
    else:
        Z1 = Normal(0,1).sample((m,))
        Z2 = Normal(0,1).sample((m,))
               
    # Normalising noise          
    Z1 -= Z1.mean()
    scale1 = ((1-R2)/R2*torch.var(Z0))**0.5
    Z1 = Z1*scale1/Z1.var()**0.5
    Z2 -= Z2.mean()
    scale2 = ((1-R2)/R2*torch.var(torch.exp(Z0+Z1)))**0.5
    Z2 = Z2*scale2/Z2.var()**0.5          
    
    # Drawing obs data
    X0 = Z0
    X1 = X0 + Z1
    X2 = torch.exp(X1)+Z2    
    X = torch.column_stack((X0,X1,X2))
    
    # Intervntional data
    if interventional_data:
        Xint = []
        
        # Getting noise
        Z0 = Normal(0,1).sample((Nint,))
        if adversarial:
            Z1 = torch.sigmoid(StudentT(2,5,0.5).sample((Nint,)))
            Z2 = torch.sigmoid(StudentT(2,5,0.5).sample((Nint,)))
        else:
            Z1 = Normal(0,1).sample((Nint,))
            Z2 = Normal(0,1).sample((Nint,))

        # Normalising noise          
        Z1 -= Z1.mean()
        scale1 = ((1-R2)/R2*torch.var(Z0))**0.5
        Z1 = Z1*scale1/Z1.var()**0.5
        Z2 -= Z2.mean()
        scale2 = ((1-R2)/R2*torch.var(torch.exp(Z0+Z1)))**0.5
        Z2 = Z2*scale2/Z2.var()**0.5 
                  
        # Drawing int data
        X0int = Z0
        for i in range(len(levels)):
            X0int = intervention_function(levels[i], X0int)
            X1int = X0int + Z1
            X2int =  torch.exp(X1int)+Z2
            Xint.append(torch.column_stack((X0int,X1int,X2int)))
    
        if scale:
            X *= 1/X.var(0)**0.5
            Xint *= 1/X.var(0)**0.5
        
        return X,Xint
    
    else:        
        if scale:
            X *= 1/X.var(0)**0.5            
        return X
                  
class simple_SCM_conditioner:
    
    def __init__(self,i):
        self.i = i
        
    def forward(self,X):
        if self.i==1:
            return X
        if self.i==2:
            return torch.exp(X)
    