import torch
from RQS import *

def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)


class Shift_Transformer:
    
    def __init__(self):
        return
        
    def forward(self,theta,y):
        return theta[0] + y
    
    def backward(self,theta,y):
        return y - theta[0]
    
    def parameters(self):
        return []
    
class Linear_Transformer:
    
    def __init__(self):
        return
        
    def forward(self,theta,y):
        return theta[0]*y
    
    def backward(self,theta,y):
        return y/theta[0]
    
    def parameters(self):
        return []
    
class Affine_Transformer:
    
    def __init__(self, log_det = False):
        self.ld = log_det
        return

    def forward(self,theta,y):
        if self.ld:
            return theta[0]+torch.exp(theta[1])*y,theta[1]          
        else:
            return theta[0]+torch.exp(theta[1])*y
    
    def backward(self,theta,y):
        if self.ld:
            return (y-theta[0])/torch.exp(theta[1]),-theta[1]
        else:
            return (y-theta[0])/torch.exp(theta[1])
    
    def parameters(self):
        return []
    
class Linear_Affine_Transformer:
    
    def __init__(self, log_det = False):
        self.ld = log_det
        return

    def forward(self,theta,y):
        if self.ld:
            return theta[0]+theta[1]*y         
        else:
            return theta[0]+theta[1]*y
    
    def backward(self,theta,y):
        if self.ld:
            return (y-theta[0])/theta[1],
        else:
            return (y-theta[0])/theta[1]
    
    def parameters(self):
        return []
    
    
class Softplus_Affine_Transformer:
    
    def __init__(self, log_det = False):
        self.ld = log_det
        return

    def forward(self,theta,y):
        if self.ld:
            return theta[0]+torch.log(1+torch.exp(theta[1]))*y,torch.log(torch.log(1+torch.exp(theta[1])))       
        else:
            return theta[0]+torch.log(1+torch.exp(theta[1]))*y
    
    def backward(self,theta,y):
        if self.ld:
            return (y-theta[0])/torch.log(1+torch.exp(theta[1])),-torch.log(torch.log(1+torch.exp(theta[1]))) 
        else:
            return (y-theta[0])/torch.log(1+torch.exp(theta[1]))
    
    def parameters(self):
        return []

class NAF_Transformer:
    
    """
    NAF transformer of the form y = inverse_sigmoid(W ^T sigmoid(AX+b))
    where W,A,b = theta  \in and are broadcastable into NxD shape if output by NN conditioners.
    """
    
    def __init__(self):
        return
    
    def forward(self,theta,y):
        
        if len(y)==1: # outer product computation for MMD est
            
            # transforming parameters
            W = torch.log(1+torch.exp(theta[0]))
            W = (W/W.sum(1)[:,None])[...,None] # N x D x 1
            A = (torch.log(1+torch.exp(theta[1])))[...,None] # N X D x 1
            b = theta[2][...,None] # N x D x 1
            y = y.T[:,None] # 1 x 1 x N
                        
        else:
        
            # transforming parameters
            W = torch.log(1+torch.exp(theta[0]))
            W = W/W.sum(1)[:,None] # N x D
            A = torch.log(1+torch.exp(theta[1])) # N X D
            b = theta[2] # N x D

        # getting features
        psi_y = torch.sigmoid(A*y + b)
            
        # returning transform
        return inv_sigmoid((W * psi_y).sum(1))[:,None] # returns N x 1 or N x N x 1 
        
    def backward(self,theta,y):
        return self.forward(theta,y)
    
    def parameters(self):
        return []
    
class RQS_Shift_Transformer:
    
    def __init__(self,widths,heights,derivatives,tail_bound = 3,
                 min_width = 1e-3, min_height = 1e-3,min_derivative = 1e-3,log_det = True):
        
        self.widths = widths
        self.heights = heights
        self.derivatives = derivatives
        self.tail_bound = tail_bound
        self.min_width = min_width
        self.min_height = min_height
        self.min_derivative = min_derivative
        self.ld = log_det

    def parameters(self):
        return [self.widths,self.heights,self.derivatives,self.tail_bound]
    
    def forward(self,theta,y):
        one = torch.ones((len(y),1))
        if self.ld:
            Tu,logdet =  unconstrained_RQS(y.view(len(y,)), 
                    one @ self.widths,
                    one @ self.heights,
                    one @ self.derivatives,
                    False,
                    self.tail_bound,
                    self.min_width,
                    self.min_height,
                    self.min_derivative,
                    self.ld)
            return Tu.view(len(y),1)+theta[0],logdet
        else:       
            Tu =  unconstrained_RQS(y.view(len(y,)), 
                    one @ self.widths,
                    one @ self.heights,
                    one @ self.derivatives,
                    False,
                    self.tail_bound,
                    self.min_width,
                    self.min_height,
                    self.min_derivative,
                    self.ld)
            return Tu.view(len(y),1)+theta[0]
    
    def backward(self,theta,y):
        one = torch.ones((len(y),1))
        if self.ld:
            u,logdet =  unconstrained_RQS((y-theta[0]).view(len(y),),
                    one @ self.widths,
                    one @ self.heights,
                    one @ self.derivatives,
                    True,
                    self.tail_bound,
                    self.min_width,
                    self.min_height,
                    self.min_derivative,
                    self.ld)
            return u.view(len(y),1),logdet
        else:
            u =  unconstrained_RQS((y-theta[0]).view(len(y),),
                    one @ self.widths,
                    one @ self.heights,
                    one @ self.derivatives,
                    True,
                    self.tail_bound,
                    self.min_width,
                    self.min_height,
                    self.min_derivative,
                    self.ld)
            return u.view(len(y),1)
    
class RQS_Softplus_Affine_Transformer:
    
    def __init__(self,widths,heights,derivatives,tail_bound = 1.0,
                 min_width = 1e-3, min_height = 1e-3,min_derivative = 1e-3,log_det = True):
        
        self.widths = widths
        self.heights = heights
        self.derivatives = derivatives
        self.tail_bound = tail_bound
        self.min_width = min_width
        self.min_height = min_height
        self.min_derivative = min_derivative
        self.ld = log_det

    def parameters(self):
        return [self.widths,self.heights,self.derivatives,self.tail_bound]
    
    def forward(self,theta,y):
        one = torch.ones((len(y),1))
        if self.ld:
            Tu,logdet =  unconstrained_RQS(y.view(len(y,)), 
                    one @ self.widths,
                    one @ self.heights,
                    one @ self.derivatives,
                    False,
                    self.tail_bound,
                    self.min_width,
                    self.min_height,
                    self.min_derivative,
                    self.ld)
            return Tu.view(len(y),1)*torch.log(1+torch.exp(theta[1]))+theta[0],logdet+torch.log(torch.log(1+torch.exp(theta[1])))
        else:       
            Tu =  unconstrained_RQS(y.view(len(y,)), 
                    one @ self.widths,
                    one @ self.heights,
                    one @ self.derivatives,
                    False,
                    self.tail_bound,
                    self.min_width,
                    self.min_height,
                    self.min_derivative,
                    self.ld)
            return Tu.view(len(y),1)*torch.log(1+torch.exp(theta[1]))+theta[0]
    
    def backward(self,theta,y):
        one = torch.ones((len(y),1))
        if self.ld:
            u,logdet =  unconstrained_RQS(((y-theta[0])/torch.log(1+torch.exp(theta[1]))).view(len(y),),
                    one @ self.widths,
                    one @ self.heights,
                    one @ self.derivatives,
                    True,
                    self.tail_bound,
                    self.min_width,
                    self.min_height,
                    self.min_derivative,
                    self.ld)
            return u.view(len(y),1),logdet-torch.log(torch.log(1+torch.exp(theta[1])))
        else:
            u =  unconstrained_RQS(((y-theta[0])/torch.log(1+torch.exp(theta[1]))).view(len(y),),
                    one @ self.widths,
                    one @ self.heights,
                    one @ self.derivatives,
                    True,
                    self.tail_bound,
                    self.min_width,
                    self.min_height,
                    self.min_derivative,
                    self.ld)
            return u.view(len(y),1)
     
    
    
    
    