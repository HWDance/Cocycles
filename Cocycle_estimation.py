# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:34:25 2023

@author: hughw
"""
import torch
import numpy as np
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
                
class train:
    
    def __init__(self,model,niter=200,learn_rate = 0.01):
        self.model = model
        self.D = model.D
        self.P = model.P
        self.niter = niter
        self.learn_rate = learn_rate
    
    def combination(self,Y,X,independent_pairs = False, double = False):
        if independent_pairs:
            n = int(len(Y)/2)
            Ytild,Xtild = ([torch.row_stack((Y[:n],Y[n:])),torch.row_stack((Y[n:],Y[:n]))],
                               [torch.row_stack((X[:n],X[n:])),torch.row_stack((X[n:],X[:n]))])
        else:
            n = len(Y)
            m = n*(n-1)
            if not double:
                m  = int(m/2)
            Ytild = (torch.zeros((m,self.P)),torch.zeros((m,self.P)))
            Xtild = (torch.zeros((m,self.D)),torch.zeros((m,self.D)))
            for d in range(self.D):
                Xcomb = torch.combinations(X[:,d])
                if double:
                    Xcomb = torch.column_stack((torch.row_stack((Xcomb[:,0],Xcomb[:,1])),
                                                torch.row_stack((Xcomb[:,1],Xcomb[:,0]))))
                Xtild[0][:,d],Xtild[1][:,d] = Xcomb[:,0],Xcomb[:,1]
            for p in range(self.P):
                Ycomb = torch.combinations(Y[:,p])
                if double:
                    Ycomb = torch.column_stack((torch.row_stack((Ycomb[:,0],Ycomb[:,1])),
                                                torch.row_stack((Ycomb[:,1],Ycomb[:,0]))))
                Ytild[0][:,p],Ytild[1][:,p] = Ycomb[:,0],Ycomb[:,1]
        
        return Ytild,Xtild
    
    def get_subsample(self,X,Y=[],subsamples=[]):
            ind_list = np.linspace(0,len(X[0])-1,len(X[0])).astype(int)
            batch_inds = torch.tensor([np.random.choice(ind_list,subsamples)]).long().view(subsamples,)
            Xbatch = [X[0][batch_inds],X[1][batch_inds]]
            if Y!=[]:
                Ybatch = [Y[0][batch_inds],Y[1][batch_inds]]
                return Xbatch,Ybatch
            else:
                return Xbatch
            
    def get_subsample_new(self,X,Y=[],subsamples=[]): # COME BACK TO FIX THIS
            ind_list = np.linspace(0,len(X[0])-1,len(X[0])).astype(int)
            batch_inds_0,batch_inds_1 = (torch.tensor([np.random.choice(ind_list,subsamples)]).long().view(subsamples,),
                                         torch.tensor([np.random.choice(ind_list,subsamples)]).long().view(subsamples,))
            Xbatch = [X[0][batch_inds_0],X[1][batch_inds_1]]
            if Y!=[]:
                Ybatch = [Y[0][batch_inds_0],Y[1][batch_inds_1]]
                return Xbatch,Ybatch
            else:
                return Xbatch

    def MMD_loss(self,X,Y,kernel_x,kernel_y, covariate_use = 2):
        # X,Y as tuples containing the two sets
        #m = len(Y)
        if covariate_use ==2:
            Xcombined = torch.column_stack((X[0],X[1]))
        elif covariate_use ==1:
            Xcombined = X[1]
        else:
            Xcombined = X[0]
        K_xx = kernel_x.get_gram(Xcombined,Xcombined)
        K_Y11 = kernel_y.get_gram(Y[1],Y[1])
        K_Y01 = kernel_y.get_gram(Y[0],Y[1])
        K_yy = K_Y11 - K_Y01 - K_Y01.T
        return torch.mean(K_xx*K_yy)
        # K_yy = (kernel_y.get_gram(Y[0],Y[0])+kernel_y.get_gram(Y[1],Y[1])-kernel_y.get_gram(Y[0],Y[1]) -kernel_y.get_gram(Y[1],Y[0]))
        #return torch.sum(torch.tril(K_xx*K_yy,0))*2/m**2
    
    def CMMD_loss(self,X,Y,kernel_x, covariate_use = 2):
        # X,Y as tuples containing the two sets
        m = len(Y)
        if covariate_use ==2:
            Xcombined = torch.column_stack((X[0],X[1]))
        elif covariate_use ==1:
            Xcombined = X[1]
        else:
            Xcombined = X[0]
        K = kernel_x.get_gram(Xcombined,Xcombined)
        Z = Y[0]-Y[1]
        return Z.T @ K @ Z/m**2
    
    def CMR_loss(self,X,Y,kernel_x,mean=0):
        # X,Y as tuples containing the two sets
        m = len(Y)
        K = kernel_x.get_gram(X,X)
        U = self.model.inverse_transformation(X,Y) - mean
        return U.T @ K @ U/m**2

    def CLS_loss(self,Y):
        m = len(Y)
        return torch.mean((Y[0]-Y[1]).abs())
        
    def LS_loss(self,X,Y, mean = 0):
        U = self.model.inverse_transformation(X,Y)
        return torch.mean((U - mean)**2)
    
    def median_heuristic(self,X):
        Dist = torch.cdist(X,X, p = 2.0)**2
        Lower_tri = torch.tril(Dist, diagonal=-1).view(len(X)**2).sort(descending = True)[0]
        Lower_tri = Lower_tri[Lower_tri!=0]
        return Lower_tri.median()

    def optimise_NN(self,Y,X, subsamples = [], subsamples_kernel = [], kernel = [], loss_fn = "GMM", loss_params = [0.0,0.0], optimise_loss_params = True, 
                        MMD_covariates = 2, train_test_split = 1, independent_combinations = False, plot = False, plot_start = 30, print_ = False, data_loader = True):
        
        # Train and Test split
        if train_test_split < 1:
            ntrain = int(train_test_split*len(Y))
            Ytrain,Xtrain = Y[:ntrain],X[:ntrain]
            Ytest,Xtest = Y[ntrain:],X[ntrain:]
        else:
            Ytrain,Xtrain = Y,X
            Ytest,Xtest = [],[]
            ntrain = len(Y)
        
        # Getting data pairs
        if loss_fn in ["GMM","CMMD", "CLS"]:
            Ytild,Xtild = self.combination(Ytrain,Xtrain, independent_combinations)
            if train_test_split <1:
                Ytild_t,Xtild_t = self.combination(Ytest,Xtest, independent_combinations)
            else:
                Ytild_t,Xtild_t = [],[]
            print("Data combination complete")
        else:
            Ytild,Xtild = [Ytrain,Ytrain],[Xtrain,Xtrain]
            Ytild_t,Xtild_t = [Ytest,Ytest],[Xtest,Xtest]
        m = len(Ytild[0])

            
        # Subsampling initialisation
        subsamples,subsamples_kernel = min(m,subsamples),min(m,subsamples_kernel)

        # Median heuristic for kernel parameters
        if loss_fn in ["GMM","CMMD", "CMR"]:
            if optimise_loss_params:
                if subsamples_kernel < m:
                    Xbatch,Ybatch = self.get_subsample(Xtild,Ytild,subsamples_kernel)
                else:
                    Xbatch,Ybatch = Xtild,Ytild
                if loss_fn in ["GMM","CMMD"] and MMD_covariates ==2:
                    Xkernel = torch.column_stack((Xbatch[0],Xbatch[1]))
                else:
                    Xkernel = Xbatch[0]
                lengthscale_x = (self.median_heuristic(Xkernel)/2).sqrt()                    
                kernel_x = kernel(lengthscale = lengthscale_x,scale = 1)
                if loss_fn == "GMM":    
                    lengthscale_y = (self.median_heuristic(Ybatch[0])/2).sqrt()
                    kernel_y = kernel(lengthscale = lengthscale_y,scale = 1)
                else:
                    lengthscale_y = loss_params[1]
                print("Median heuristic complete, lengthscales are : ", lengthscale_x,lengthscale_y)
            else:
                kernel_x = kernel(lengthscale = loss_params[0],scale = 1)
                kernel_y = kernel(lengthscale = loss_params[1],scale = 1)            
        print("Starting optimisation")
        
        # Dataloader
        if data_loader:
            loader = DataLoader(list(zip(Xtild[0],Xtild[1],Ytild[0],Ytild[1])), shuffle=True, batch_size=subsamples)
            
        # Optimisation
        params_list = []
        for k in range(len(self.model.conditioner)):
            params_list +=  self.model.conditioner[k].parameters()
        moment_param = torch.tensor([loss_params[0]],requires_grad = optimise_loss_params)
        if optimise_loss_params and loss_fn == "CMR":
            params_list += [moment_param]

        optimizer = torch.optim.Adam(params_list, lr=self.learn_rate)  
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
        Losses = torch.zeros(self.niter)
        for i in range(self.niter):
            optimizer.zero_grad()
            #Subsampling
            if subsamples < m:
                Xbatch,Ybatch = self.get_subsample(Xtild,Ytild,subsamples)
                if train_test_split < 1:
                    Xbatch_t,Ybatch_t = self.get_subsample(Xtild_t,Ytild_t,subsamples)
            elif i==0:
                Xbatch,Ybatch,Xbatch_t,Ybatch_t  = Xtild,Ytild,Xtild_t,Ytild_t
            # Getting loss ( EVENTUALLY NEED TO MOVE THIS INTO ITS OWN MODULE)
            if loss_fn in ["GMM","CMMD", "CLS"]:
                Zbatch = (Ybatch[0],self.model.cocycle(Xbatch[0],Xbatch[1],Ybatch[1]))
                if train_test_split < 1:
                    Zbatch_t = (Ybatch_t[0],self.model.cocycle(Xbatch_t[0],Xbatch_t[1],Ybatch_t[1]))
            if loss_fn == "GMM":
                loss = self.MMD_loss(Xbatch,Zbatch,kernel_x,kernel_y, MMD_covariates)
                if train_test_split < 1:
                    loss_t = self.MMD_loss(Xbatch_t,Zbatch_t,kernel_x,kernel_y,MMD_covariates)
            elif loss_fn == "CMMD":
                loss = self.CMMD_loss(Xbatch,Zbatch,kernel_x,MMD_covariates)
                if train_test_split < 1:
                    loss_t = self.CMMD_loss(Xbatch_t,Zbatch_t,kernel_x,MMD_covariates)
            elif loss_fn == "CMR":
                loss = self.CMR_loss(Xbatch[0],Ybatch[0],kernel_x, moment_param)
                if train_test_split < 1:
                    loss_t = self.CMR_loss(Xbatch_t[0],Ybatch_t[0],kernel_x, moment_param)
            elif loss_fn == "CLS":
                loss = self.CLS_loss(Zbatch)
                if train_test_split < 1:
                    loss_t = self.CLS_loss(Zbatch_t)
            elif loss_fn == "LS":
                loss = self.LS_loss(Xbatch[0],Ybatch[0],moment_param)                
                if train_test_split < 1:
                    loss_t = self.LS_loss(Xbatch_t[0],Ybatch_t[0],moment_param) 
            # Optimisation step
            loss.backward()
            optimizer.step()
            # Display
            if print_:
                if train_test_split < 1:
                    Losses[i] = loss_t.detach()
                else:
                    Losses[i] = loss.detach()   
                if not i % 10:
                    clear_output(wait=True)
                    print("Loss last 10 avg is :",Losses[i-10:i].mean()) #, self.model.conditioner[0].state_dict())
                    print("Completion % :", (i+1)/self.niter*100)
                    if plot and i > plot_start:
                        if optimise_loss_params and loss_fn == "CMR":    
                            print("Loss mean is : ", moment_param)
                        plt.plot(Losses[20:i+1])
                        display(plt.gcf())
        if loss_fn in ["GMM","CMMD","CMR"] :
            print("Final lengthscales are:", lengthscale_x)
        return self.model,Losses