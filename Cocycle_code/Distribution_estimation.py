# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:13:25 2023

@author: hughw
"""
import torch
import os
from torch.distributions import Normal
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Distribution:
    
    def __init__(self):
        return
    
    def sample(self):
        return

class CV:
    
    def __init__(folds=5):
        self.folds = folds
    
    def get_CV_splits(self,X):
        """
        X: nxd matrix
        folds: # CV folds 

        Returns: list of folds x training and validation sets
        """
    
        n = len(X)
        if self.folds==[]:
            self.folds = n
        n_per_fold = int(n/self.folds+1-1e-10) # rounds up except if exact integer
        row_count = torch.linspace(0,n-1,n) 
        train_val_sets = list()

        for i in range(self.folds):
            test_inds = ((row_count>= n_per_fold*i)*(row_count<n_per_fold*(i+1)))>0
            train_inds = test_inds==0
            train_val_sets.append([X[train_inds],X[test_inds]])
    
        return train_val_sets
    
    def get_CV_loss(self,X,Y,functional):
        loss = 0
        for i in range(self.folds):
            Xtrain,Xtest = X[i][0],X[i][1]
            Ytrain,Ytest = Y[i][0],Y[i][1]
            Ypred = functional(Ytrain,Xtrain,Xtest)
            loss += torch.mean((Ytest - Ypred)**2)/self.folds     

        return loss
    
    
class Conditional_Expectation_Regressor(CV):
    
    """
    Instantiates a regression function x -> f(x) =  E[Y|x]
    """
    
    def __init__(self, functional):
        self.functional = functional
        self.hyperparameters = self.functional.hyperparameters
    
    def get_weights(self,Ytrain,Xtrain):
        self.weights = self.functional.get_weights(Ytrain,Xtrain)
    
    def get_features(self,Xtrain):
        self.features =  self.functional.get_features(Xtrain)
    
    def forward(self,Ytrain,Xtrain,Xtest):
        return self.functional(Ytrain,Xtrain,Xtest)
    
    def __call__(self,X):
        return self.features(X) @ self.weights
    
    def optimise(self,X,Y,miniter=100,maxiter = 1000,nfold = 5, learn_rate = 0.01, tol = 1e-4):
                  
        # Getting data folds
        self.folds = nfold
        Xsplits = self.get_CV_splits(X)
        Ysplits = self.get_CV_splits(Y)

        # doing optimisation
        optimizer = torch.optim.Adam(self.hyperparameters, lr=learn_rate)  
        Losses = torch.zeros(maxiter)
        i=0
        while (i < miniter) or (i< maxiter and Losses[i-11:i-1].mean() - Losses[:i-49].min() < - tol):
            optimizer.zero_grad()
            loss = self.get_CV_loss(Xsplits,Ysplits,self.functional)
            loss.backward()
            optimizer.step()
            Losses[i] = loss.detach()
            if not i % 10:
                print("iter", i, ", loss = ", Losses[i])   
            i += 1
        return Losses  
        
        
        
    
    
    

