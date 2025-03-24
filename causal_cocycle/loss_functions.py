import torch
from torch.distributions import Normal,Uniform
import numpy as np 

class Loss:
    
    def __init__(self,loss_fn, kernel = [], outer_subsample = []):
        """
        loss_fn : str matching loss functions below
        kernel : list of up to two kernels for inputs and outputs respectively
        """
        self.loss_fn = loss_fn  # String e.g. "CMMD_M"
        self.kernel = kernel
        self.outer_subsample = outer_subsample
        
    def get_subsample(self,inputs,outputs=[],subsamples=[]):
            ind_list = np.linspace(0,len(inputs)-1,len(inputs)).astype(int)
            batch_inds = torch.tensor([np.random.choice(ind_list,subsamples)]).long().view(subsamples,)
            inputs_batch = inputs[batch_inds]
            if outputs!=[]:
                outputs_batch = outputs[batch_inds]
                return inputs_batch,outputs_batch
            else:
                return inputs_batch,[]
                
    def median_heuristic(self,inputs,outputs = [],subsamples = 10000):
        """
        Returns median heuristic lengthscale for Gaussian kernel
        """
        
        # Subsampling
        if subsamples < len(inputs):
                inputs_batch,outputs_batch = self.get_subsample(inputs,outputs,subsamples)
        else:
            inputs_batch,outputs_batch = inputs,outputs
        
        # Median heurstic for inputs
        Dist = torch.cdist(inputs_batch,inputs_batch, p = 2.0)**2
        Lower_tri = torch.tril(Dist, diagonal=-1)
        mask = torch.tril(torch.ones_like(Lower_tri, dtype=torch.bool), -1)
        self.kernel[0].lengthscale =  (Lower_tri[mask].median()/2).sqrt()
        
        # Median heuristic for outputs
        if outputs != []:
            Dist = torch.cdist(outputs_batch,outputs_batch, p = 2.0)**2
            Lower_tri = torch.tril(Dist, diagonal=-1)
            mask = torch.tril(torch.ones_like(Lower_tri, dtype=torch.bool), -1)
            self.kernel[1].lengthscale =  (Lower_tri[mask].median()/2).sqrt()
            
        return
    
    def HSIC(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """         
        # Getting dimensions 
        m = len(outputs)
        
        # Making prediction
        U = model.inverse_transformation(inputs,outputs)
        
        # Getting gram matrix
        K_xx = self.kernel[0].get_gram(inputs,inputs)
        K_uu = self.kernel[1].get_gram(U,U)
        
        # Getting centering matrix
        H = torch.eye(m) - 1/m*torch.ones((m,m))
        
        return torch.trace(K_xx @ H @ K_uu @ H)/m**2 
    
    
    def CMMD_V(self,model,inputs,outputs): # no input features - this method pulls outer sum outside the norm
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)
        
        # Make model prediction
        outputs_pred = model.cocycle_outer(inputs,inputs,outputs)
        if len(outputs_pred.size())<3:
            outputs_pred = outputs_pred[...,None] # adding extra dimension to make sure output is N x N x 1 here

        # Computing kernels in batches to avoid memory overload
        K = 0
        if n**3 >= 10**8:
            batchsize = max(1,min(n,int(10**8/n**2)))
        else:
            batchsize = n
        nbatch = int(n/batchsize)
        for i in range(nbatch):
            # Get gram matrices
            K += self.kernel[1].get_gram(outputs_pred[i*batchsize:(i+1)*batchsize],outputs_pred[i*batchsize:(i+1)*batchsize]).sum()/n**3
            K += -2*self.kernel[1].get_gram(outputs[i*batchsize:(i+1)*batchsize,None,:],outputs_pred[i*batchsize:(i+1)*batchsize]).sum()/n**2
        
        return K
    
    def CMMD_U(self,model,inputs,outputs): # no input features - this method pulls outer sum outside the norm
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)

        # Getting mask
        Mask = torch.ones((n,n,n))
        for i in range(n):
                Mask[i,:,i] = 0
                Mask[i,i,:] = 0
                Mask[:,i,i] = 0        
        
        # Make model prediction
        outputs_pred = model.cocycle_outer(inputs,inputs,outputs)
        if len(outputs_pred.size())<3:
            outputs_pred = outputs_pred[...,None] # adding extra dimension to make sure output is N x N x 1 here

        # Get gram matrices
        K1 = self.kernel[1].get_gram(outputs_pred,outputs_pred) # N x N x N now 
        K2 = -2*self.kernel[1].get_gram(outputs_pred,outputs[:,None,:])[...,0] # N x N
        
        return (K1*Mask).sum()/(n*(n-1)*(n-2)) + (K2*(1-torch.eye(n))).sum()/(n*(n-1))   
    
    def URR(self,model,inputs,outputs):
        Dist =  Normal(0,1)
        output_samples_1 = model.transformation(inputs,Dist.sample((len(inputs),1)))
        output_samples_2 = model.transformation(inputs,Dist.sample((len(inputs),1)))
        K1 = self.kernel[1].get_gram(output_samples_1[...,None],output_samples_2[...,None])
        K2 = self.kernel[1].get_gram(output_samples_1[...,None],outputs[...,None])
        return (K1 - 2*K2).mean()

    def LS(self,model,inputs,outputs):
        return (model.inverse_transformation(inputs,outputs).abs()**2).mean()
    
    def __call__(self,model,inputs,outputs):
        """
        Returns a function L : (model,inputs,outputs) -> loss
        """
        if self.loss_fn == "HSIC":
            return self.HSIC(model,inputs,outputs)
        if self.loss_fn == "CMMD_V":
            return self.CMMD_V(model,inputs,outputs)
        if self.loss_fn == "CMMD_U":
            return self.CMMD_U(model,inputs,outputs)
        if self.loss_fn == "URR":
            return self.URR(model,inputs,outputs)
        if self.loss_fn == "LS":
            return self.LS(model,inputs,outputs)
        
        
        
    
        
    
        
