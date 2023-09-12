import torch
from RQS import *

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
        return [self.widths,self.heights,self.derivatives]
    
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
     
    
    
    
    