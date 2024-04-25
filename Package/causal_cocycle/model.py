# -*- coding: utf-8 -*-
"""
@author: hughw

Need to fix code so conditioner only calls forward once when computing cocycle
"""
import torch

class cocycle_outcome_model:

    def __init__(self,model,inputs_train,outputs_train):

        """
        model: cocycle model object
        inputs_train: inputs trained on (N x D torch.tensor)
        outputs_train: outputs trained on (N x P torch.tensor)
        """
        self.model = model
        self.inputs = inputs_train
        self.outputs = outputs_train

    def __call__(self,inputs,feature):
        """
        estimates E[f(Y)|X]: (N x D -> K x N x P)

        inputs: (N x D) torch.tensor
        featurs: function (N x P -> K x N x P)
        """
        prediction = self.model.cocycle_outer(inputs,self.inputs,self.outputs) # M x N x P
        return feature(prediction).mean(2)
        
class flow_outcome_model:

    def __init__(self,model,noise_samples):

        """
        model: flow model object
        noise_samples: noise samples from base dist (N x P torch.tensor)
        """
        self.model = model
        self.noise = noise_samples

    def __call__(self,inputs,feature):
        """
        estimates E[f(Y)|X]: (N x D -> K x N x P)

        inputs: (N x D) torch.tensor
        featurs: function (N x P -> K x N x P)
        """
        prediction = self.model.transformation_outer(inputs,self.noise) # M x N x P
        return feature(prediction).mean(2)
        

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
            eye_y = torch.ones((len(y),1))
            eye_x = torch.ones((len(x),1))
            outer_parameters = torch.kron(self.conditioner[i].forward(x),eye_y)
            outer_y = torch.kron(eye_x,y)
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
    