
import torch 
from torch.distributions import Normal,Laplace,LogNormal,Beta,StudentT,Uniform,Gamma

def Simpson_Nonlin(N = 1000, Nint = 10**6,interventional_data = False, intervention = [], levels = [], noise_dists = [[]*4],transforms = [[lambda x : x]*4]):
    
    # Total samples to draw 
    m = N + Nint    
    
    # Drawing noise
    Z = torch.zeros((m,4))
    for i in range(4):
        if noise_dists[i] == []:
            Z[:,i] = Normal(0,1).sample((m,))
        else:
            Z[:,i] = transforms[i](noise_dists[i].sample((m,)))
    
    # Drawing observational data
    X = torch.zeros((m,4))
    X[:,0] = -Z[:,0]
    U1 = torch.exp(1-X[:,0])
    U1[torch.isinf(U1)] = 1e+30
    X[:,1] = torch.log(1+U1)+(3/20)**0.5*Z[:,1]
    X[:,2] = torch.tanh(2*X[:,1]) + 3/2*X[:,0] -1 + Z[:,2]
    X[:,3] = 5*torch.tanh((X[:,2]-4)/5) + 3  + 1/10**0.5*Z[:,3]
    
    # Drawing interventional data
    if interventional_data:
        Xintlist = []
        for i in range(len(levels)):
            Xint = torch.zeros((Nint,4))
            Xint[:,0] = -Z[N:,0]
            U1 = torch.exp(1-Xint[:,0])
            U1[torch.isinf(U1)] = 1e+30
            Xint[:,1] = intervention(levels[i],torch.log(1+U1)+(3/20)**0.5*Z[N:,1])
            Xint[:,2] = torch.tanh(2*Xint[:,1]) + 3/2*Xint[:,0] -1 + Z[N:,2]
            Xint[:,3] = 5*torch.tanh((Xint[:,2]-4)/5) + 3  + 1/10**0.5*Z[N:,3]
            Xintlist.append(Xint)
        
        return X,Xintlist
                                        
    else:                                
        return X

class Simpson_conditioner:
    
    def __init__(self):
        self.parents = [[],[0],[0,1],[2]]
        
    def forward(self,X):
        if i==0:
            return X*0
        if i==1:
            return torch.log(1+torch.exp(1-X))
        if i==2:
            return torch.tanh(2*X[:,1].view(len(X),1)) + 3/2*X[:,0].view(len(X),1) -1
        if i==3:
            return 5*torch.tanh((X-4)/5) + 3
    
def Fork_Nonlin(N = 1000, Nint = 10**6,interventional_data = False, intervention = [], levels = [], noise_dists = [[]*4],transforms = [[lambda x : x]*4]):
    
    # Total samples to draw 
    m = N + Nint    
    
    # Drawing noise
    Z = torch.zeros((m,4))
    for i in range(4):
        if noise_dists[i] == []:
            Z[:,i] = Normal(0,1).sample((m,))
        else:
            Z[:,i] = transforms[i](noise_dists[i].sample((m,)))
    
    # Drawing observational data
    X = torch.zeros((m,4))                           
    X[:,0] = Z[:,0]
    X[:,1] = -Z[:,1]
    X[:,2] = 4/(1+torch.exp(-X[:,0]-X[:,1])) - X[:,1]**2 + 0.5*Z[:,2]
    X[:,3] = 20/(1+torch.exp(0.5*X[:,2]**2 - X[:,2])) + Z[:,3]
    
    # Drawing interventional data
    if interventional_data:
        Xintlist = []
        for i in range(len(levels)):
            Xint = torch.zeros((Nint,4))
            Xint[:,0] = Z[N:,0]
            Xint[:,1] = intervention(levels[i],-Z[N:,1])
            Xint[:,2] = 4/(1+torch.exp(-Xint[:,0]-Xint[:,1])) - Xint[:,1]**2 + 0.5*Z[N:,2]
            Xint[:,3] = 20/(1+torch.exp(0.5*Xint[:,2]**2 - Xint[:,2])) + Z[N:,3]
            Xintlist.append(Xint)
        
        return X,Xintlist
                                        
    else:                                
        return X
    
class Fork_conditioner:
    
    def __init__(self):
        self.parents = [[],[],[0,1],[2]]
        
    def forward(self,X):
        if i==0 or i==1:
            return X*0
        if i==2:
            return 4/(1+torch.exp(-X[:,0]-X[:,1])) - X[:,1]**2
        if i==3:
            return 20/(1+torch.exp(0.5*X**2 - X))
    
def Nonlin_Gauss_chain(N = 1000, Nint = 10**6,interventional_data = False, intervention = [], levels = [], noise_dists = [[]*4],transforms = [[lambda x : x]*3]):
    # Total samples to draw
    m = N + Nint
    beta = 1-6*(1/5**0.5-1/3) # noise scaling
    
    # Drawing noise
    Z = torch.zeros((m,3))
    for i in range(3):
        if noise_dists[i] == []:
            Z[:,i] = Normal(0,1).sample((m,))
        else:
            Z[:,i] = transforms[i](noise_dists[i].sample((m,)))
    
    # Drawing observational data
    X = torch.zeros((m,3))
    X[:,0] = Z[:,0]
    X[:,1] = X[:,0] -1 + Z[:,1]
    X[:,2] = 6**0.5*torch.exp(-X[:,1]**2)+beta*Z[:,2]    
    
    # Drawing interventional data
    if interventional_data:
        Xintlist = []
        for i in range(len(levels)):
            Xint = torch.zeros((Nint,3))
            Xint[:,0] = Z[N:,0]
            Xint[:,1] = intervention(levels[i],Xint[:,0] -1 + Z[N:,1])
            Xint[:,2] = 6**0.5*torch.exp(-Xint[:,1]**2)+beta*Z[N:,2]    
            Xintlist.append(Xint)
        
        return X,Xintlist
                                        
    else:                                
        return X
    
class NonlinGausschain_conditioner:
    
    def __init__(self):
        self.parents = [[],[0],[1]]
        
    def forward(self,X):
        if i==0:
            return X*0
        if i==1:
            return X - 1
        if i==2:
            return 6**0.5*torch.exp(-X**2)
        
def Nonlin_Gauss_dense(N = 1000, Nint = 10**6,interventional_data = False, intervention = [], levels = [], noise_dists = [[]*4],transforms = [[lambda x : x]*3]):
    # Total samples to draw
    m = N + Nint
    beta = 1-6*(1/5**0.5-1/3) # noise scaling
    
    # Drawing noise
    Z = torch.zeros((m,3))
    for i in range(3):
        if noise_dists[i] == []:
            Z[:,i] = Normal(0,1).sample((m,))
        else:
            Z[:,i] = transforms[i](noise_dists[i].sample((m,)))
    
    # Drawing observational data
    X = torch.zeros((m,3))
    X[:,0] = Z[:,0]
    X[:,1] = X[:,0] -1 + Z[:,1]
    X[:,2] = 6**0.5*torch.exp(-X[:,1]**2-X[:,0]**2)+beta*Z[:,2]    
    
    # Drawing interventional data
    if interventional_data:
        Xintlist = []
        for i in range(len(levels)):
            Xint = torch.zeros((Nint,3))
            Xint[:,0] = Z[N:,0]
            Xint[:,1] = intervention(levels[i],Xint[:,0] -1 + Z[N:,1])
            Xint[:,2] = 6**0.5*torch.exp(-Xint[:,1]**2-Xint[:,0]**2)+beta*Z[N:,2]    
            Xintlist.append(Xint)
        
        return X,Xintlist
                                        
    else:                                
        return X
    
class NonlinGaussdense_conditioner:
    
    def __init__(self):
        self.parents = [[],[0],[0,1]]
        
    def forward(self,X,i):
        if i==0:
            return X*0
        if i==1:
            return X - 1
        if i==2:
            return 6**0.5*torch.exp(-X[:,0]*2-X[:,1]*2)