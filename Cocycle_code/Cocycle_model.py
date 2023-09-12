# -*- coding: utf-8 -*-
"""
@author: hughw

Need to fix code to conditioner only calls forward once when computing cocycle
"""

class cocycle_model:
    
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
            l.append(self.conditioner[i].forward(x))
        return self.transformer.backward(l,y)
    
    def cocycle(self,x1,x2,y):
        return self.transformation(x1,self.inverse_transformation(x2,y))
    