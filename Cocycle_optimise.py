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
    
    def __init__(self,model):
        self.model = model
    
    def get_subsample(self,inputs,outputs=[],subsamples=[]):
            ind_list = np.linspace(0,len(inputs)-1,len(inputs)).astype(int)
            batch_inds = torch.tensor([np.random.choice(ind_list,subsamples)]).long().view(subsamples,)
            inputs_batch = inputs[batch_inds]
            if outputs!=[]:
                outputs_batch = outputs[batch_inds]
                return inputs_batch,outputs_batch
            else:
                return inputs_batch

    def optimise(self,loss,inputs,outputs, inputs_val = [],outputs_val = [],optimiser = "adam", learn_rate = 0.001, maxiter = 2000,miniter = 100, optimise_loss_params = True, 
                 val_loss = True, val_tol = 1e-3, batch_size = 1024, val_batch_size = 4096,batch_sampling = "random", scheduler = False, schedule_milestone = 100, lr_mult = 0.9, 
                     plot = False, plot_start = 30, print_ = False):
        
        # Dimensions
        m = len(outputs)
        m_val = len(outputs_val)
        
        # Parameters set up
        params_list = []
        for k in range(len(self.model.conditioner)):
            params_list +=  self.model.conditioner[k].parameters()
        loss_params = torch.tensor([loss.parameters],requires_grad = optimise_loss_params)
        if optimise_loss_params and loss.loss_fn in ["CMR","LS"]:
            params_list += [loss_params]
        
        # Optimiser set up
        if optimiser == "adam":
            optimizer = torch.optim.Adam(params_list, lr=learn_rate)  
        else:
            optimizer = torch.optim.SGD(params_list, lr=learn_rate)   
        if scheduler:
            lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[schedule_milestone]*50, gamma=lr_mult)
        
        # Optimisation iterations
        Losses = torch.zeros(maxiter)
        if val_loss:
            Losses_val = torch.zeros(maxiter)
        batch_start,batch_end = 0,batch_size
        i = 0
        while i < miniter or (i < maxiter and (not val_loss or Losses_val[i-10:i-1].mean() - Losses_val[:i-99].min() < val_tol)):
            optimizer.zero_grad()
            #Subsampling
            if batch_size < m and batch_sampling == "random":
                inputs_batch,outputs_batch = self.get_subsample(inputs,outputs,batch_size)
            elif batch_size < m:
                if batch_end < m:
                    batch_start += batch_size
                    batch_end += batch_size
                else:
                    batch_start,batch_end = 0,batch_size
                if batch_end > m:
                    batch_end = m
                inputs_batch,outputs_batch = inputs[batch_start:batch_end],outputs[batch_start:batch_end]
            else:
                inputs_batch,outputs_batch = inputs,outputs

            # Loss computation
            Loss = loss(self.model,inputs_batch,outputs_batch)
            Losses[i] = Loss.detach()
            if inputs_val != [] and val_loss: 
                if val_batch_size < m_val:
                    inputs_batch_val,outputs_batch_val = self.get_subsample(inputs_val,outputs_val,batch_size)
                else:
                    inputs_batch_val,outputs_batch_val = inputs_val,outputs_val
                Loss_val = loss(self.model,inputs_batch_val,outputs_batch_val)
                Losses_val[i] = Loss_val.detach()

            # Optimisation step
            Loss.backward()
            optimizer.step()

            # Display
            if print_ or plot:
                if not i % 10:
                    clear_output(wait=True)
                    if print_:
                        print("Training loss last 10 avg is :",Losses[i-10:i].mean())
                        if val_loss:
                            print("Validation loss last 10 avg is :",Losses_val[i-10:i].mean())
                        print("Completion % :", (i+1)/maxiter*100)
                    if plot and i > plot_start:
                        plt.plot(Losses[20:i+1])
                        if val_loss and inputs_val != []:
                            plt.plot(Losses_val[20:i+1])
                        display(plt.gcf())
            i += 1
       
        return self.model,params_list