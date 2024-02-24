import torch
from torch.distributions import Normal,Uniform
import numpy as np 

class Loss:
    
    def __init__(self,loss_fn, kernel = [], kernel_covariates = 1, outer_subsample = [],features = [],get_CMMD_mask = False,mask_size = 128):
        """
        loss_fn : str matching loss functions below
        kernel : list of up to two kernels for inputs and outputs respectively
        kernel_covariates : (for cocycle MMD) 0 = (X), 1 = (X,X'), 2 = (X,X',Y')
        """
        self.loss_fn = loss_fn  # String e.g. "CMMD_M"
        self.kernel = kernel
        self.outer_subsample = outer_subsample
        self.features = features
        if get_CMMD_mask:
            Mask = torch.ones((mask_size,
                               mask_size,
                               mask_size))
            for i in range(mask_size):
                    Mask[i,:,i] = 0
                    Mask[i,i,:] = 0
                    Mask[:,i,i] = 0
            self.mask = Mask
            
        
    def get_RFF_features(self,features):
        self.A = Normal(0,1/self.kernel[1].lengthscale).sample((features,)).view((features,))
        self.b = Uniform(0,2*torch.acos(torch.zeros(1)).item()).sample((features,))
        
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
        Lower_tri = torch.tril(Dist, diagonal=-1).view(len(inputs_batch)**2).sort(descending = True)[0]
        Lower_tri = Lower_tri[Lower_tri!=0]
        self.kernel[0].lengthscale =  (Lower_tri.median()/2).sqrt()
        
        # Median heuristic for outputs
        if outputs != []:
            Dist = torch.cdist(outputs_batch,outputs_batch, p = 2.0)**2
            Lower_tri = torch.tril(Dist, diagonal=-1).view(len(outputs_batch)**2).sort(descending = True)[0]
            Lower_tri = Lower_tri[Lower_tri!=0]
            self.kernel[1].lengthscale =  (Lower_tri.median()/2).sqrt()
            
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
        #U = model.inverse_transformation(inputs,outputs).T
        outputs_pred = model.cocycle_outer(inputs,inputs,outputs)
        if len(outputs_pred.size())<3:
            outputs_pred = outputs_pred[...,None] # adding extra dimension to make sure output is N x N x 1 here
        
        if not self.outer_subsample:
            nrows_sample = max(min(n,int(10**9/n**2)),1) # To prevent memory overload, subsample from outer sum
        else:
            nrows_sample = max(min(self.outer_subsample,int(10**9/self.outer_subsample**2)),1) # To prevent memory overload, subsample from outer sum            
        if nrows_sample < n:
            outputs_pred_row_batch,outputs_row_batch = self.get_subsample(outputs_pred,outputs,subsamples = nrows_sample) # nrow x N x 1 tensor
        else:
            outputs_pred_row_batch,outputs_row_batch = outputs_pred,outputs
            
        
        # Get gram matrices
        K = self.kernel[1].get_gram(outputs_pred_row_batch,outputs_pred_row_batch).mean()
        K += -2*self.kernel[1].get_gram(outputs_row_batch[:,None,:],outputs_pred_row_batch).mean()
        
        return K
    
    def CMMD_U(self,model,inputs,outputs): # no input features - this method pulls outer sum outside the norm
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)
        
        # Make model prediction
        #U = model.inverse_transformation(inputs,outputs).T
        outputs_pred = model.cocycle_outer(inputs,inputs,outputs)
        if len(outputs_pred.size())<3:
            outputs_pred = outputs_pred[...,None] # adding extra dimension to make sure output is N x N x 1 here
        
        if not self.outer_subsample:
            nrows_sample = max(min(n,int(10**9/n**2)),1) # To prevent memory overload, subsample from outer sum
        else:
            nrows_sample = max(min(self.outer_subsample,int(10**9/self.outer_subsample**2)),1) # To prevent memory overload, subsample from outer sum            
        if nrows_sample < n:
            outputs_pred_row_batch,outputs_row_batch = self.get_subsample(outputs_pred,outputs,subsamples = nrows_sample) # nrow x N x 1 tensor
        else:
            outputs_pred_row_batch,outputs_row_batch = outputs_pred,outputs
            
        
        # Get gram matrices
        K1 = self.kernel[1].get_gram(outputs_pred_row_batch,outputs_pred_row_batch) # N x N x N now 
        K2 = -2*self.kernel[1].get_gram(outputs_pred_row_batch,outputs_row_batch[:,None,:])[...,0] # N x N
        
        
        return (K1*self.mask).sum()/(n*(n-1)*(n-2)) + (K2*(1-torch.eye(n))).sum()/(n*(n-1))   
    
    def URR(self,model,inputs,outputs):
        Dist =  Normal(0,1)
        output_samples_1 = model.transformation(inputs,Dist.sample((len(inputs),1)))
        output_samples_2 = model.transformation(inputs,Dist.sample((len(inputs),1)))
        K1 = self.kernel[1].get_gram(output_samples_1[...,None],output_samples_2[...,None])
        K2 = self.kernel[1].get_gram(output_samples_1[...,None],outputs[...,None])
        return (K1 - 2*K2).mean()
    
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
        elif self.loss_fn == "URR":
            return self.URR(model,inputs,outputs)
        
        
    
        
    
        
