import torch
from RQS import *

def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)

class Transformer:
    """
    Aggregrate transformer that takes in monotone map layers
    """
    
    def __init__(self,layers,logdet=False):
        self.layers = layers
        self.logdet = logdet
        
    def forward(self,theta,y):
        logdet = torch.zeros(len(y))
        for i in range(len(self.layers)):
            y,ld = self.layers[len(self.layers)-(i+1)].forward(theta[len(self.layers)-(i+1)],y,self.logdet)
            logdet += ld
        if self.logdet:
            return y,logdet
        else:
            return y
        
    def backward(self,theta,y):  
        logdet = torch.zeros(len(y))
        for i in range(len(self.layers)):
            y,ld = self.layers[i].backward(theta[i],y,self.logdet)
            logdet += ld
        if self.logdet:
            return y,logdet
        else:
            return y
           
class Shift_layer:
    """
    g : (theta,y) -> theta + y
    """
    
    def __init__(self,transform = lambda x : x):
        self.transform = transform
    
    def forward(self,theta,y,logdet=False):
        ld = 0
        return self.transform(theta) + y,ld
    
    def backward(self,theta,y,logdet=False):
        ld = 0
        return y - self.transform(theta),ld
    
class Scale_layer:
    """
    g : (theta,y) -> theta * y
    """
    
    def __init__(self,transform = lambda x : torch.log(1+torch.exp(x))):
        self.transform = transform
        
    def forward(self,theta,y,logdet=False):
        if logdet:
            ld = torch.log(self.transform(theta[0]))
        else:
            ld = 0
        return self.transform(theta)*y,ld
    
    def backward(self,theta,y,logdet=False):
        if logdet:
            ld = -torch.log(self.transform(theta[0]))
        else:
            ld = 0
        return y/self.transform(theta),ld
    
class Inverse_layer:
    
    """
    g : (theta,y) -> theta/y
    """
    
    def __init__(self):
        return
    
    def forward(self,theta,y,logdet=False):
        if logdet:
            ld = torch.log(theta)-2*torch.log(y)
        else:
            ld = 0
        return theta/y,ld
    
    def backward(self,theta,y,logdet=False):
        if logdet:
            ld = torch.log(theta)-2*torch.log(y)       
        else:
            ld = 0
        return theta/y,ld
    
class Hyperbolic_RELU_layer:
    """
    g : (knot,y) -> y if y >= knot and knot +1/knot - 1/y if y < knot
    
    Used to map R_+ or R_- to R in forward transformation
    """
    
    def __init__(self, knot, domain_flip = False):
        self.knot = knot
        self.flip = domain_flip
        return
    
    def forward(self,theta,y,logdet=False): # no free parameters
        f1 = y
        f2 = (self.knot + 1/self.knot - 1/y)
        
        if logdet:
            ld = (1*(y>=self.knot) + (1/y**2)*(y<self.knot))
        else:
            ld = 0
        return (-1)**self.flip*(f1*(y>=self.knot) + f2*(y<self.knot)),ld
    
    def backward(self,theta,y,logdet=False): # no free parameters
        f1inv = y
        f2inv = (1/(self.knot + 1/self.knot - y))
        
        if logdet:
            ld = (1*((-1)**self.flip*y>=self.knot) + (f2inv)**2*((-1)**self.flip*y<self.knot))
        else:
            ld = 0
            
        return f1inv*((-1)**self.flip*y>=self.knot) + (-1)**self.flip*f2inv*((-1)**self.flip*y<self.knot),ld
    
class RQS_layer:
    """
    g: (theta,y) -> RQS_theta(y)
    """
    
    def __init__(self,bins=8,min_width = 1e-3, min_height = 1e-3,min_derivative = 1e-3):
        
        self.min_width = min_width
        self.min_height = min_height
        self.min_derivative = min_derivative
        self.bins = bins
        self.inputs_in_mask_itercount = 0
        
    def forward(self,theta,y,logdet = False):
        Tu,ld,in_mask_count =  unconstrained_RQS(y.view(len(y),), 
                                    theta[:,:self.bins],
                                    theta[:,self.bins:2*self.bins],
                                    theta[:,2*self.bins:(3*self.bins+1)],
                                    False,
                                    theta[:,-1].abs().mean(), # NEED TO CHANGE THIS TO DEPEND ON ALL INPUTS
                                    self.min_width,
                                    self.min_height,
                                    self.min_derivative,
                                    logdet)
        self.inputs_in_mask_itercount += in_mask_count
        return Tu.view(len(y),1),ld
    
    def backward(self,theta,y,logdet = False):
        u,ld,in_mask_count =  unconstrained_RQS(y.view(len(y),),
                                    theta[:,:self.bins],
                                    theta[:,self.bins:2*self.bins],
                                    theta[:,2*self.bins:(3*self.bins+1)],
                                    True,
                                    theta[:,-1].abs().mean(), # NEED TO CHANGE THIS TO DEPEND ON ALL INPUTS
                                    self.min_width,
                                    self.min_height,
                                    self.min_derivative,
                                    logdet)
        self.inputs_in_mask_itercount += in_mask_count
        return u.view(len(y),1),ld
    