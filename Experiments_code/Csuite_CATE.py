
import torch 
from torch.distributions import Normal,Laplace,LogNormal,Beta,StudentT,Uniform,Gamma

def Simpson_Nonlin(N = 1000, Nint = 10**6,interventional_data = False, intervention_function = [], int_levels = [], adversarial = False,scale = False, alpha = 1, cate = False,cate_levels = []):
    m = N + Nint
    if adversarial:
        Z0 = -(Gamma(alpha,alpha**0.5).sample((m,)) -alpha**0.5)
        Z1 = (Gamma(alpha,alpha**0.5).sample((m,)) -alpha**0.5)
    else:    
        Z0 = Normal(0,1).sample((m,))
        Z1 = Normal(0,1).sample((m,))
    Z2 = Normal(0,1).sample((m,))
    Z3 = Normal(0,1).sample((m,))
    
    if cate:
        X = []
        for c in range(cate_levels):
            X0 = Z0*0 + cate_levels[c]
            X1 = torch.log(1+torch.exp(1-X0))+(3/20)**0.5*Z1
            X2 = torch.tanh(2*X1) + 3/2*X0 -1 + Z2
            X3 = 5*torch.tanh((X2-4)/5) + 3  + 1/10**0.5*Z3
            X.append(torch.column_stack((X0,X1,X2,X3)))            
    else:
        X0 = Z0
        X1 = torch.log(1+torch.exp(1-X0))+(3/20)**0.5*Z1
        X2 = torch.tanh(2*X1) + 3/2*X0 -1 + Z2
        X3 = 5*torch.tanh((X2-4)/5) + 3  + 1/10**0.5*Z3
        X = torch.column_stack((X0,X1,X2,X3))
    
    if interventional_data:
        if adversarial:
            Z0 = -(Gamma(alpha,alpha**0.5).sample((Nint,)) -alpha**0.5)
            Z1 = (Gamma(alpha,alpha**0.5).sample((Nint,)) -alpha**0.5)
        else:    
            Z0 = Normal(0,1).sample((Nint,))
            Z1 = Normal(0,1).sample((Nint,))
        Z2 = Normal(0,1).sample((Nint,))
        Z3 = Normal(0,1).sample((Nint,))
        
        if cate:
            Xint = []
            for c in range(len(cate_levels)):
                Xint_c = []
                X0int = Z0*0+cate_levels[c]
                X1int = torch.log(1+torch.exp(1-X0int))+(3/20)**0.5*Z1
                for i in range(len(int_levels)):
                    X1int = intervention_function(int_levels[i],X1int)
                    X2int = torch.tanh(2*X1int) + 3/2*X0int -1 + Z2
                    X3int = 5*torch.tanh((X2int-4)/5) + 3  + 1/10**0.5*Z3
                    Xint_c.append(torch.column_stack((X0int,X1int,X2int,X3int)))
                Xint.append(Xint_c)      
        else:
            Xint = []
            X0int = Z0
            X1int = torch.log(1+torch.exp(1-X0int))+(3/20)**0.5*Z1

            for i in range(len(int_levels)):
                X1int = intervention_function(int_levels[i],X1int)
                X2int = torch.tanh(2*X1int) + 3/2*X0int -1 + Z2
                X3int = 5*torch.tanh((X2int-4)/5) + 3  + 1/10**0.5*Z3
                Xint.append(torch.column_stack((X0int,X1int,X2int,X3int)))
        
        if scale:
            X *= 1/X.var(0)**0.5
            Xint *= 1/X.var(0)**0.5
        
        return X,Xint
    else:        
        if scale:
            X *= 1/X.var(0)**0.5
            Xint *= 1/X.var(0)**0.5
            
        return X

class Simpson_conditioner:
    
    def __init__(self,i):
        self.i = i
        
    def forward(self,X):
        if self.i==1:
            return torch.log(1+torch.exp(1-X))
        if self.i==2:
            return torch.tanh(2*X[:,1].view(len(X),1)) + 3/2*X[:,0].view(len(X),1)
        if self.i==3:
            return 5*torch.tanh((X-4)/5) + 3
    
def Fork_Nonlin(N = 1000, Nint = 10**6,interventional_data = False, intervention_function = [], int_levels = [], adversarial = False,scale = False,alpha = 1, cate = False,cate_levels = []):
    m = N + Nint
    if adversarial:
        Z0 = Gamma(alpha,alpha**0.5).sample((m,))-alpha**0.5
        Z1 =-(Gamma(alpha,alpha**0.5).sample((m,))-alpha**0.5)
    else:    
        Z0 = Normal(0,1).sample((m,))
        Z1 = Normal(0,1).sample((m,))
    Z2 = Normal(0,1).sample((m,))
    Z3 = Normal(0,1).sample((m,))
    
    if cate:
        X = []
        for c in range(len(cate_levels)):
            X0 = Z0*0 + cate_levels[c]
            X1 = Z1
            X2 = 4/(1+torch.exp(-X0-X1)) - X1**2 + 0.5*Z2
            X3 = 20/(1+torch.exp(0.5*X2**2 - X2)) + Z3
            X.append(torch.column_stack((X0,X1,X2,X3)))
    else:
        X0 = Z0
        X1 = Z1
        X2 = 4/(1+torch.exp(-X0-X1)) - X1**2 + 0.5*Z2
        X3 = 20/(1+torch.exp(0.5*X2**2 - X2)) + Z3
        X = torch.column_stack((X0,X1,X2,X3))
    
    if interventional_data:
        if adversarial:
            Z0 = Gamma(alpha,alpha**0.5).sample((Nint,))-alpha**0.5
            Z1 =-(Gamma(alpha,alpha**0.5).sample((Nint,))-alpha**0.5)
        else:    
            Z0 = Normal(0,1).sample((Nint,))
            Z1 = Normal(0,1).sample((Nint,))
        Z2 = Normal(0,1).sample((Nint,))
        Z3 = Normal(0,1).sample((Nint,))
        
        if cate:
            Xint = []
            for c in range(len(cate_levels)):
                Xint_c = [] 
                X0int = Z0*0 + cate_levels[c]
                X1int = Z1

                for i in range(len(int_levels)):
                    X1int = intervention_function(int_levels[i],X1int)
                    X2int = 4/(1+torch.exp(-X0int-X1int)) - X1int**2 + 0.5*Z2
                    X3int = 20/(1+torch.exp(0.5*X2int**2 - X2int)) + Z3
                    Xint_c.append(torch.column_stack((X0int,X1int,X2int,X3int)))
                Xint.append(Xint_c)
                    
        else:
            Xint = []
            X0int = Z0
            X1int = Z1

            for i in range(len(int_levels)):
                X1int = intervention_function(int_levels[i],X1int)
                X2int = 4/(1+torch.exp(-X0int-X1int)) - X1int**2 + 0.5*Z2
                X3int = 20/(1+torch.exp(0.5*X2int**2 - X2int)) + Z3
                Xint.append(torch.column_stack((X0int,X1int,X2int,X3int)))
        
        if scale:
            X *= 1/X.var(0)**0.5
            Xint *= 1/X.var(0)**0.5
        
        return X,Xint
    else:        
        if scale:
            X *= 1/X.var(0)**0.5
            Xint *= 1/X.var(0)**0.5
            
        return X   
    
class Fork_conditioner:
    
    def __init__(self,i):
        self.i = i
        
    def forward(self,X):
        if self.i==2:
            return 4/(1+torch.exp(-X[:,0]-X[:,1])) - X[:,1]**2
        if self.i==3:
            return 20/(1+torch.exp(0.5*X**2 - X))
    
def Nonlin_Gauss_chain(N = 1000, Nint = 10**6,interventional_data = False, intervention_function = [], int_levels = [], adversarial = False, scale = False, alpha = 1, cate = False,cate_levels = []):
    
    m = N + Nint
    beta = 1-6*(1/5**0.5-1/3)
    if adversarial:
        Z0 = Gamma(alpha,alpha**0.5).sample((m,))-alpha**0.5
        Z1 = Gamma(alpha,alpha**0.5).sample((m,))-alpha**0.5
    else:    
        Z0 = Normal(0,1).sample((m,))
        Z1 = Normal(0,1).sample((m,))
    Z2 = Normal(0,1).sample((m,))
    if cate:
        X = []
        for c in range(len(cate_levels)):
            X0 = Z0*0 + cate_levels[c]
            X1 = X0 -1 + Z1
            X2 = 6**0.5*torch.exp(-X1**2)+beta*Z2    
            X.append(torch.column_stack((X0,X1,X2)))              

    else:
        X0 = Z0
        X1 = X0 -1 + Z1
        X2 = 6**0.5*torch.exp(-X1**2)+beta*Z2    
        X = torch.column_stack((X0,X1,X2))
    
    if interventional_data:
        if adversarial:
            Z0 = Gamma(alpha,alpha**0.5).sample((Nint,))-alpha**0.5
            Z1 = Gamma(alpha,alpha**0.5).sample((Nint,))-alpha**0.5
        else:    
            Z0 = Normal(0,1).sample((Nint,))
            Z1 = Normal(0,1).sample((Nint,))
        Z2 = Normal(0,1).sample((Nint,))
        if cate:
            Xint = []
            for c in range(len(cate_levels)):
                Xint_c = []
                X0int = Z0*0 + cate_levels[c]
                X1int = X0int - 1 + Z1
                for i in range(len(int_levels)):
                    X1int = intervention_function(int_levels[i], X1int)
                    X2int =  6**0.5*torch.exp(-X1int**2)+beta*Z2
                    Xint_c.append(torch.column_stack((X0int,X1int,X2int)))
                Xint.append(Xint_c)           
        else:
            Xint = []
            X0int = Z0
            X1int = X0int - 1 + Z1
            for i in range(len(int_levels)):
                X1int = intervention_function(int_levels[i], X1int)
                X2int =  6**0.5*torch.exp(-X1int**2)+beta*Z2
                Xint.append(torch.column_stack((X0int,X1int,X2int)))

        if scale:
            X *= 1/X.var(0)**0.5
            Xint *= 1/X.var(0)**0.5
        
        return X,Xint
    
    else:        
        if scale:
            X *= 1/X.var(0)**0.5            
        return X
    
class NonlinGausschain_conditioner:
    
    def __init__(self,i):
        self.i = i
        
    def forward(self,X):
        if self.i==1:
            return X - 1
        if self.i==2:
            return 6**0.5*torch.exp(-X**2)
        
    