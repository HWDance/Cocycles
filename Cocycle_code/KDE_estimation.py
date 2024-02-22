# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:13:25 2023

@author: hughw
"""
import torch
import os
from torch.distributions import Normal
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class KDE:
    
    def __init__(self,kernel, weights = []):
        self.kernel = kernel
        self.weights = weights
        
    def pi(self):
        return torch.acos(torch.zeros(1)).item() * 2

    def forward(self,X,centers):
        if self.weights ==[]:
            self.weights = 1/len(centers)
        K = self.kernel.get_gram(X,centers)
        return (K*self.weights).sum(1)
    
    def get_CV_splits(self,X,folds):
        """
        X: nxd matrix
        folds: # CV folds 

        Returns: list of folds x training and validation sets
        """
    
        n = len(X)
        if folds==[]:
            folds = n
        n_per_fold = int(n/folds+1-1e-10) # rounds up except if exact integer
        row_count = torch.linspace(0,n-1,n) 
        train_val_sets = list()

        for i in range(folds):
            test_inds = ((row_count>= n_per_fold*i)*(row_count<n_per_fold*(i+1)))>0
            train_inds = test_inds==0
            train_val_sets.append([X[train_inds],X[test_inds]])
    
        return train_val_sets

    def get_KDE_CV_loss(self,X,folds,reg,Loss):
        loss = 0
        for i in range(folds):
            Xtrain,Xtest = X[i][0],X[i][1]
            if Loss == "loglik":
                loss += -torch.sum(torch.log(self.forward(Xtest,Xtrain)+reg))
            else:
                K_test_train = self.kernel.get_gram(Xtest[:,1:],Xtrain[:,1:])
                ypred = K_test_train @ Xtrain[:,:1] / K_test_train.sum(1)[:,None]
                
                loss += torch.mean((Xtest[:,:1] - ypred)**2) 
                

        return loss
    
    def normalise_scale(self,reg,Loss,input_dims):
        if len(self.kernel.lengthscale.size())==1:
                S = torch.eye(input_dims)*self.kernel.lengthscale
        else:
                S = torch.ones((input_dims,input_dims))*self.kernel.lengthscale
        if Loss == "loglik":
                logdet = torch.logdet(S.T @ S +torch.eye(input_dims)*reg)
        self.kernel.scale = (2*self.pi())**-(input_dims/2) * torch.exp(logdet)**0.5
        return
        
        

    def optimise(self,X,learn_rate,miniter,maxiter,tol,nfold,reg = 1e-10, Loss = "loglik"): # Currently compatible with MVN kernel only
                
        # Getting data folds
        Xsplits = self.get_CV_splits(X,nfold)
        d = len(X.T)

        # doing optimisation
        ls = torch.zeros(d,requires_grad = True) # lengthscale initialisation
        optimizer = torch.optim.Adam([ls], lr=learn_rate)  
        Losses = torch.zeros(maxiter)
        logdet = 0
        i = 0
        while (i < miniter) or (i< maxiter and Losses[i-1] - Losses[i-11] < - tol):
            optimizer.zero_grad()
            self.kernel.lengthscale = torch.exp(ls)
            self.normalise_scale(reg,Loss,input_dims = d)
            loss = self.get_KDE_CV_loss(Xsplits,nfold,reg,Loss)
            loss.backward()
            optimizer.step()
            Losses[i] = loss.detach()
            if not i % 10:
                print("iter", i, ", loss = ", Losses[i])   
            i += 1
            
        # Final hyper update
        self.kernel.lengthscale = torch.exp(ls)
        self.normalise_scale(reg,Loss, input_dims = d)
        return Losses  
        
    def sample(self,X, nsamples=10**4,reg=1e-5):
        # Sampling X
        d = len(X.T)
        Z = X[torch.randint(0,len(X),(nsamples,))]
        
        # Sampling Gaussian noise
        U = Normal(0,1).sample((nsamples,len(X.T)))
        if len(self.kernel.lengthscale.size())==1:
            S = torch.eye(d)*self.kernel.lengthscale
        else:
            S = torch.ones((d,d))*self.kernel.lengthscale
        C_half = torch.sqrt(torch.linalg.inv(S.T @ S+torch.eye(d)*reg))
        S = U @ C_half
        
        return Z+S