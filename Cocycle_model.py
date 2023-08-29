# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:34:25 2023

@author: hughw
"""

class cocycle_model:
    
    def __init__(self,conditioner,transformer,inverse_transformer):
        self.conditioner = conditioner # list of functions f: X -> Th 
        self.transformer = transformer # g: Th x Y -> Y 
        self.inverse_transformer = inverse_transformer # g^-1: Th x Y -> Y 
    
    def transformation(self,x,y):
        l = []
        for i in range(len(self.conditioner)):
            l.append(self.conditioner[i].forward(x))
        return self.transformer(*l,y)
    
    def inverse_transformation(self,x,y):
        l = []
        for i in range(len(self.conditioner)):
            l.append(self.conditioner[i].forward(x))
        return self.inverse_transformer(*l,y)
    
    def cocycle(self,x1,x2,y):
        return self.transformation(x1,self.inverse_transformation(x2,y))