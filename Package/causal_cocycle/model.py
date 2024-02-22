# -*- coding: utf-8 -*-
"""
@author: hughw

Need to fix code so conditioner only calls forward once when computing cocycle
"""
import torch

class cocycle_model:
    
    def __init__(self,conditioner,transformer):
        self.conditioner = conditioner # list of model classes that outputs transformer inputs
        self.transformer = transformer # bijective transformer
    
    def transformation(self,x,y):
        transformer_parameters = []
        for i in range(len(self.conditioner)):
            transformer_parameters.append(self.conditioner[i].forward(x))
        return self.transformer.forward(transformer_parameters,y)
    
    def transformation_outer(self,x,y):
        transformer_parameters = []
        for i in range(len(self.conditioner)):
            eye = torch.ones((len(y),1))
            outer_parameters = torch.kron(self.conditioner[i].forward(x),eye)
            outer_y = torch.kron(eye,y)
            transformer_parameters.append(outer_parameters)
        return self.transformer.forward(transformer_parameters,outer_y).reshape(len(x),len(y))
    
    def inverse_transformation(self,x,y):
        transformer_parameters = []
        for i in range(len(self.conditioner)):
            transformer_parameters.append(self.conditioner[i].forward(x))
        return self.transformer.backward(transformer_parameters,y)
    
    def cocycle(self,x1,x2,y):
        return self.transformation(x1,self.inverse_transformation(x2,y)) 
    
    def cocycle_outer(self,x1,x2,y):
        """
        returns (len(x1) = len(x2)) x len(y) matrix
        """
        return self.transformation_outer(x1,self.inverse_transformation(x2,y))

class flow_model:
    """
    Same as cocycle model, but includes flexibility to return log-determinants
    """
    
    def __init__(self,conditioner,transformer):
        self.conditioner = conditioner # list of model classes that outputs transformer inputs
        self.transformer = transformer # bijective transformer
    
    def transformation(self,x,y):
        transformer_parameters = []
        for i in range(len(self.conditioner)):
            transformer_parameters.append(self.conditioner[i].forward(x))
        return self.transformer.forward(transformer_parameters,y)
    
    def transformation_outer(self,x,y):
        transformer_parameters = []
        for i in range(len(self.conditioner)):
            eye = torch.ones((len(y),1))
            outer_parameters = torch.kron(self.conditioner[i].forward(x),eye)
            outer_y = torch.kron(eye,y)
            transformer_parameters.append(outer_parameters)
        return self.transformer.forward(transformer_parameters,outer_y).reshape(len(x),len(y))
    
    def inverse_transformation(self,x,y):
        transformer_parameters = []
        for i in range(len(self.conditioner)):
            transformer_parameters.append(self.conditioner[i].forward(x))
        return self.transformer.backward(transformer_parameters,y)
    
    def cocycle(self,x1,x2,y):
        return self.transformation(x1,self.inverse_transformation(x2,y)) 
    
    def cocycle_outer(self,x1,x2,y):
        """
        returns (len(x1) = len(x2)) x len(y) matrix
        """
        return self.transformation_outer(x1,self.inverse_transformation(x2,y))

    
    
class bb_cocycle_model:
    
    def __init__(self,conditioner,transformer):
        self.conditioner = conditioner # list of model classes that outputs transformer inputs
        self.transformer = transformer # bijective transformer
    
    def transformation(self,x,y):
        l = []
        for i in range(len(self.conditioner)):
            l.append(self.conditioner[i].forward(x))
        return self.transformer.forward(l,y)
    
    def inverse_transformation(self,x,y):
        l = []
        for i in range(len(self.conditioner)):
            l.append(self.conditioner[i].backward(x))
        return self.transformer.backward(l,y)
    
    def cocycle(self,x1,x2,y):
        return self.transformation(x1,self.inverse_transformation(x2,y))
    
    