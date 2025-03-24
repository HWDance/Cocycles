import torch
from torch import nn
from torch.distributions import Normal,Uniform

class Composite_Conditioner:
    
    def __init__(self, conditioners):
        self.conditioners = conditioners
        self.params = []
        for i in range(len(self.conditioners)):
            self.params.append(self.conditioners[i].parameters())
            
    def parameters(self):
        return self.params
    
    def forward(self,X):
        outputs =  torch.zeros((len(X),0))
        for i in range(len(self.conditioners)):
            outputs = torch.column_stack((outputs,
                                          self.conditioners[i].forward(X)))
    
class Empty_Conditioner:
    
    def __init__(self):
        self.params = torch.ones(1,requires_grad = True)
    
    def parameters(self):
        return [self.params]
    
    def forward(self,X):
        return torch.zeros((len(X),1))
    
class Constant_Conditioner_1D:
    
    def __init__(self,init = 1.0):
        self.params = init.requires_grad_(True)
    
    def parameters(self):
        return [self.params]
    
    def forward(self,X):
        return self.params
    
class Constant_Conditioner:
    
    def __init__(self,init = 1.0, full = True, grad = True):
        self.params = init.requires_grad_(grad)
        self.full = full
    
    def parameters(self):
        return [self.params]
    
    def forward(self,X):
        if self.full:
            return torch.ones(len(X),1) @ self.params
        else:
            return self.params
    
class Constant_Conditioner_FB:
    
    def __init__(self, dims = 10):
        self.dims = dims
        self.params = Normal(0,1).sample((2*self.dims,1)).requires_grad_(True)
    
    def parameters(self):
        return [self.params]
    
    def forward(self,X):
        return self.params[:self.dims].T
    
    def backward(self,X):
        return self.params[self.dims:].T

class Lin_Conditioner(nn.Module):
    def __init__(self,d,p,bias=True, init = [], grad=True):
        super().__init__()
        self.flatten = nn.Flatten()
        lin_layer = nn.Linear(d,p,bias)
        if init!=[]:
            lin_layer.weight = nn.Parameter(
                torch.full((p, d), init), 
                requires_grad=grad
            )

        self.stack = nn.Sequential(
           lin_layer
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.stack(x)
        return y

class RFF_Conditioner:
    
    def __init__(self,features,input_dims,sd=1):
        self.params = [Normal(0,sd/features).sample((features,1)).requires_grad_(True), # weights
                       Normal(0,sd/input_dims).sample((features,input_dims)).requires_grad_(True), # feature scale
                       Uniform(0,2*pi).sample((features,1)).requires_grad_(True), # feature shift
                       torch.ones((input_dims,),requires_grad = True)] # input dim relevance
    
    def parameters(self):
        return self.params

    def forward(self, X : "N x D"):
        
        return (self.params[0].T @ torch.cos(self.params[1] @ (X*self.params[3]).T +self.params[2])).T
    
class NN_RELU_Conditioner(nn.Module):
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
        fx = self.stack(x)
        return fx
    
class NN_RELU_Manifold_Conditioner(nn.Module):
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True,manifold_dim = 1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            *([nn.Linear(input_dims, manifold_dim),nn.Linear(manifold_dim, width)]+
              [nn.ReLU(),nn.Linear(width, width)]*(layers-1)+
              [nn.ReLU(), nn.Linear(width, output_dims, bias = bias)]),
        )

    def forward(self, x):
        x = self.flatten(x)
        fx = self.stack(x)
        return fx
    
class NN_Tanh_Conditioner(nn.Module):
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            *([nn.Linear(input_dims, width)]+
              [nn.Tanh(),nn.Linear(width, width)]*(layers-1)+
              [nn.Tanh(), nn.Linear(width, output_dims, bias = bias)]),
        )

    def forward(self, x):
        x = self.flatten(x)
        fx = self.stack(x)
        return fx

    
class NN_Tanh_Manifold_Conditioner(nn.Module):
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True,manifold_dim = 1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            *([nn.Linear(input_dims, manifold_dim),nn.Linear(manifold_dim, width)]+
              [nn.Tanh(),nn.Linear(width, width)]*(layers-1)+
              [nn.Tanh(), nn.Linear(width, output_dims, bias = bias)]),
        )

    def forward(self, x):
        x = self.flatten(x)
        fx = self.stack(x)
        return fx
    
class NNdrop_Conditioner(nn.Module):
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True,p=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            *([nn.Linear(input_dims, width)]+
              [nn.ReLU(),nn.Dropout(p),nn.Linear(width, width)]*(layers-1)+
              [nn.ReLU(), nn.Linear(width, output_dims, bias = bias)]),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.stack(x)
        return y

class NN_Conditioner_FB(nn.Module):
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack_forward = nn.Sequential(
            *([nn.Linear(input_dims, width)]+
              [nn.ReLU(),nn.Linear(width, width)]*(layers-1)+
              [nn.ReLU(), nn.Linear(width, output_dims, bias = bias)]),
        )
        self.stack_backward = nn.Sequential(
            *([nn.Linear(input_dims, width)]+
              [nn.ReLU(),nn.Linear(width, width)]*(layers-1)+
              [nn.ReLU(), nn.Linear(width, output_dims, bias = bias)]),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        y = self.stack_forward(x)
        return y
    
    def backward(self, x):
        x = self.flatten(x)
        y = self.stack_backward(x)
        return y


class Lin_Conditioner_T(nn.Module):
    """
    returns class dependent conditioner - needs class variable to be listed first
    check to ensure class variable X \in \{0,1,...K\}
    """
    def __init__(self,input_dims,output_dims = 1,bias = True, init = [],classes = 2):
        super().__init__()
        self.classes = classes
        self.input_dims = output_dims
        self.output_dims = output_dims
        self.flatten = nn.Flatten()
        self.stack = []
        for i in range(classes):
            lin_layer = nn.Linear(self.input_dims,self.output_dims,bias)
            if init!=[]:
                lin_layer.weight = nn.Parameter(init,
                                            requires_grad = True)
            self.stack += [nn.Sequential(lin_layer)]

        
    def forward(self, x):
        class_variable = x[:,:1]
        inputs = x[:,1:]
        y = torch.zeros((len(inputs),self.output_dims))
        for i in range(self.classes):
            inds = torch.where(class_variable==i)[0]
            y[inds] = self.stack[i](self.flatten(inputs[inds]))
        return y

class NN_RELU_Conditioner_T(nn.Module):
    """
    returns class dependent conditioner - needs class variable to be listed first
    check to ensure class variable X \in \{0,1,...K\}
    """
    def __init__(self,width,layers,input_dims,output_dims = 1,bias = True, classes = 2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = [nn.Sequential(
            *([nn.Linear(input_dims, width)]+
              [nn.ReLU(),nn.Linear(width, width)]*(layers-1)+
              [nn.ReLU(), nn.Linear(width, output_dims, bias = bias)]),
        ) for i in range(classes)]
        self.classes = classes
        self.output_dims = output_dims
        
    def forward(self, x):
        class_variable = x[:,:1]
        inputs = x[:,1:]
        y = torch.zeros((len(inputs),self.output_dims))
        for i in range(self.classes):
            inds = torch.where(class_variable==i)[0]
            y[inds] = self.stack[i](self.flatten(inputs[inds]))
        return y
                    
pi = torch.acos(torch.zeros(1)).item() * 2 