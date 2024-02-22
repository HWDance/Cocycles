import torch
from torch.distributions import Normal,Uniform
import numpy as np 

class Loss:
    
    def __init__(self,loss_fn, loss_param = 1.0, kernel = [], kernel_covariates = 1, outer_subsample = [],features = [], lmbda = 1):
        """
        loss_fn : LS (least squares), 
                  CLS (cocycles least squares), 
                  CLS_M (cocycle least squares with mean subtract), 
                  CMR (conditional moment restrictions), 
                  CMR_M (conditional moment restrictions with mean subtract)
                  CMMD (cocycle conditional mean kernel MMD), 
                  JMMD (cocycle joint kernel MMD)
        loss_param : free parameter of loss function (e.g. mean of moment conditions)
        kernel : list of up to two kernels for inputs and outputs respectively
        kernel_covariates : (for cocycle MMD) 0 = (X), 1 = (X,X'), 2 = (X,X',Y')
        """
        self.loss_fn = loss_fn
        self.parameters = loss_param
        self.kernel = kernel
        self.kernel_covariates = kernel_covariates
        self.outer_subsample = outer_subsample
        self.features = features
        self.lmbda = lmbda
        
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
    
    def HSIC_uncentered(self,model,inputs,outputs):
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
        
        return torch.trace(K_xx @ K_uu @ H)/m**2 
    
    def JMMD(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : (X,X',Y') , n x 2d+p
        outputs : Y' , n x p
        """
        # Getting dimensions
        d = int((len(inputs.T)-1)/2)
        
        # Make model prediction
        outputs_pred = model.cocycle(inputs[:,:d],inputs[:,d:2*d],inputs[:,2*d:])
        
        # Stack inputs together
        covariates_inds = [d,2*d,2*d+1]
        max_ind = covariates_inds[self.kernel_covariates]
        
        # Get gram matrices 
        K_xx = self.kernel[0].get_gram(inputs[:,:max_ind],inputs[:,:max_ind])
        K_Y11 = self.kernel[0].get_gram(outputs_pred,outputs_pred)
        K_Y01 = self.kernel[1].get_gram(outputs_pred,outputs)
        K_yy = K_Y11 - K_Y01 - K_Y01.T
        
        return torch.mean(K_xx*K_yy)
    
    def JMMD_M(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)
        
        # Make model prediction
        U = model.inverse_transformation(inputs,outputs).T
        outputs_pred = model.transformation(inputs,U)[:,:,None]   # N x N  x 1 tensor (need to check if works for multivariate Y) 
        
        nrows_sample = max(min(n,int(10**9/n**2)),1) # To prevent memory overload, subsample from outer sum
        if nrows_sample < n:
            outputs_pred_row_batch,outputs_row_batch = self.get_subsample(outputs_pred,outputs,subsamples = nrows_sample) # nrow x N x 1 tensor
        else:
            outputs_pred_row_batch,outputs_row_batch = outputs_pred,outputs
            
        # Get gram matrices
        K_xx = self.kernel[0].get_gram(inputs,inputs)
        K = 0
        one = torch.ones((nrows_sample,1,1))
        for i in range(nrows_sample):
            K += (K_xx[None,:]*self.kernel[1].get_gram(one*outputs_pred_row_batch[i][None,:,:],outputs_pred_row_batch)).mean() # returns (N x N x N).mean()
            K += -2*(K_xx*self.kernel[1].get_gram(outputs_row_batch,outputs_pred_row_batch[i])).mean() #(N x N).mean()
        
        return K*n
    
    def JMMD_M_RFF(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)
        # Make model prediction
        U = model.inverse_transformation(inputs,outputs).T
        outputs_pred = model.transformation(inputs,U) 
        features_pred = torch.cos(torch.einsum('p,qr->pqr', self.A, outputs_pred)+self.b[:,None,None]).mean(2) # returns d x N matrix
        features_pred -= torch.cos(self.A.view(len(self.A),1)*outputs.view(1,n) + self.b[:,None])
        
        # Getting gram matrix
        K_xx = self.kernel[0].get_gram(inputs,inputs)

        return (K_xx * (features_pred.T @ features_pred)).mean()
    
    def JMMD_M_features(self,model,inputs,outputs):
        return
    
    def CMMD(self,model,inputs,outputs): # no input features - this method pulls both sums outside the norm
        return
    
    def CMMD_M(self,model,inputs,outputs): # no input features - this method pulls outer sum outside the norm
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)
        
        # Make model prediction
        U = model.inverse_transformation(inputs,outputs).T
        outputs_pred = model.transformation(inputs,U)
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
    
    def CMMD_M_reg(self,model,inputs,outputs,lmbda = 1): # no input features - this method pulls outer sum outside the norm
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)
        
        # Make model prediction
        U = model.inverse_transformation(inputs,outputs).T
        outputs_pred = model.transformation(inputs,U)
        if len(outputs_pred.size())<3:
            outputs_pred = outputs_pred[...,None] # adding extra dimension to make sure output is N x N x 1 here
        
        nrows_sample = max(min(n,int(10**9/n**2)),1) # To prevent memory overload, subsample from outer sum
        if nrows_sample < n:
            outputs_pred_row_batch,outputs_row_batch = self.get_subsample(outputs_pred,outputs,subsamples = nrows_sample) # nrow x N x 1 tensor
        else:
            outputs_pred_row_batch,outputs_row_batch = outputs_pred,outputs
            
        
        # Get gram matrices
        K = self.kernel[1].get_gram(outputs_pred_row_batch,outputs_pred_row_batch).mean()
        K += -2*self.kernel[1].get_gram(outputs_row_batch[:,None,:],outputs_pred_row_batch).mean()
        
        return K + self.lmbda*torch.mean((model.cocycle(inputs,inputs,outputs)-outputs)**2)**0.5
    
    def CMMD_M_RFF(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)
        # Make model prediction
        U = model.inverse_transformation(inputs,outputs).T
        outputs_pred = model.transformation(inputs,U) 
        features_pred = torch.cos(torch.einsum('p,qr->pqr', self.A, outputs_pred)+self.b[:,None,None]).mean(2) # returns d x N matrix
        features_pred -= torch.cos(self.A.view(len(self.A),1)*outputs.view(1,n) + self.b[:,None])
        return ((features_pred)**2).mean()
    
    def CMMD_M_features(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """
        
        # Dimensions
        n = len(outputs)
        # Make model prediction
        U = model.inverse_transformation(inputs,outputs).T
        outputs_pred = model.transformation(inputs,U) 
        features = self.features(outputs)[...,-1]  # returns d x N matrix
        features_pred = self.features(outputs_pred).mean(2) # returns d x N matrix
        return (((features - features_pred)**2)/features.var(1)[:,None]).mean()

    def CMMD_old(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : (X,X',Y') , n x 2d+p
        outputs : Y' , n x p
        """      
        # Getting dimensions
        d = int((len(inputs.T)-1)/2)
        m = len(outputs)
        
        # Make model prediction
        outputs_pred = model.cocycle(inputs[:,:d],inputs[:,d:2*d],inputs[:,2*d:])
        
        # Stack inputs together
        covariates_inds = [d,2*d,2*d+1]
        max_ind = covariates_inds[self.kernel_covariates]
        
        # Get gram matrix
        K_xx = self.kernel[0].get_gram(inputs[:,:max_ind],inputs[:,:max_ind])
        
        Z = outputs_pred-outputs
        return Z.T @ K_xx @ Z/m**2
    
    def CMR(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """          
        # Getting dimensions 
        m = len(outputs)
        
        # Making prediction
        U = model.inverse_transformation(inputs,outputs) - self.parameters
        
        # Getting gram matrix
        K_xx = self.kernel[0].get_gram(inputs,inputs)
        
        return U.T @ K_xx @ U/m
    
    def CMR_M(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """          
        # Getting dimensions 
        m = len(outputs)
        
        # Making prediction
        Umean = model.inverse_transformation(inputs,outputs).mean()
        outputs_pred = model.transformation(inputs,Umean)
        
        # Getting gram matrix
        K_xx = self.kernel[0].get_gram(inputs,inputs)
        
        Z = outputs_pred-outputs
        return Z.T @ K_xx @ Z/m

    def CLS(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : (X,X',Y') , n x 2d+p
        outputs : Y' , n x p
        """           
        # Getting dimensions
        d = int((len(inputs.T)-1)/2)
        
        # Making prediction
        outputs_pred = model.cocycle(inputs[:,:d],inputs[:,d:2*d],inputs[:,2*d:])
        
        return torch.mean((outputs - outputs_pred)**2)
    
    def CLS_M(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y' , n x p
        """           
        # Getting dimensions
        d = int((len(inputs.T)-1)/2)
        
        # Making prediction
        Umean = model.inverse_transformation(inputs,outputs).mean()
        outputs_pred = model.transformation(inputs,Umean)
        
        return torch.mean((outputs - outputs_pred)**2)
        
    def L1(self,model,inputs,outputs):
        """
        model: cocycle_model
        inputs : X , n x d
        outputs : Y , n x p
        """          
        # Making prediction
        U = model.inverse_transformation(inputs,outputs)
        
        return torch.mean(U.abs())
    
    def MLE(self,model,inputs,outputs):
        U,logdet = model.inverse_transformation(inputs,outputs)
        return torch.mean(U**2 - 2*logdet)
    
    def URR(self,model,inputs,outputs):
        Dist =  Normal(0,self.parameters**2)
        output_samples_1 = model.transformation(inputs,Dist.sample((len(inputs),1)))
        output_samples_2 = model.transformation(inputs,Dist.sample((len(inputs),1)))
        K1 = self.kernel[1].get_gram(output_samples_1[...,None],output_samples_2[...,None])
        K2 = self.kernel[1].get_gram(output_samples_1[...,None],outputs[...,None])
        return (K1 - 2*K2).mean()
    
    def URR_loglik(self,model,inputs,outputs):
        Dist =  Normal(model.transformation(inputs,outputs*0),self.parameters**2)
        output_samples_1 = Dist.sample()
        output_samples_2 = Dist.sample()
        K1 = self.kernel[1].get_gram(output_samples_1[...,None],output_samples_2[...,None])
        K2 = self.kernel[1].get_gram(output_samples_1[...,None],outputs[...,None])
        loglik = Dist.log_prob(output_samples_1)
        return (loglik*(K1-2*K2)).mean()
    
    def __call__(self,model,inputs,outputs):
        """
        Returns a function L : (model,inputs,outputs) -> loss
        """
        if self.loss_fn == "HSIC":
            return self.HSIC(model,inputs,outputs)
        if self.loss_fn == "HSIC_uncentered":
            return self.HSIC_uncentered(model,inputs,outputs)
        if self.loss_fn == "JMMD":
            return self.JMMD(model,inputs,outputs)
        if self.loss_fn == "JMMD_M":
            return self.JMMD_M(model,inputs,outputs)
        if self.loss_fn == "JMMD_M_RFF":
            return self.JMMD_M_RFF(model,inputs,outputs)
        if self.loss_fn == "JMMD_M_features":
            return self.JMMD_M_features(model,inputs,outputs)
        if self.loss_fn == "CMMD":
            return self.CMMD(model,inputs,outputs)
        if self.loss_fn == "CMMD_M":
            return self.CMMD_M(model,inputs,outputs)
        if self.loss_fn == "CMMD_M_reg":
            return self.CMMD_M_reg(model,inputs,outputs)
        if self.loss_fn == "CMMD_M_RFF":
            return self.CMMD_M_RFF(model,inputs,outputs)
        if self.loss_fn == "CMMD_M_features":
            return self.CMMD_M_features(model,inputs,outputs)
        elif self.loss_fn == "CMMD":
            return self.CMMD(model,inputs,outputs)
        elif self.loss_fn == "CMR":
            return self.CMR(model,inputs,outputs)
        elif self.loss_fn == "CMR_M":
            return self.CMR_M(model,inputs,outputs)
        elif self.loss_fn == "CLS":
            return self.CLS(model,inputs,outputs)
        elif self.loss_fn == "CLS_M":
            return self.CLS_M(model,inputs,outputs)
        elif self.loss_fn == "L1":
            return self.L1(model,inputs,outputs)
        elif self.loss_fn == "MLE":
            return self.MLE(model,inputs,outputs)
        elif self.loss_fn == "URR":
            return self.URR(model,inputs,outputs)
        elif self.loss_fn == "URR_loglik":
            return self.URR_loglik(model,inputs,outputs)
        
        
    
        
    
        
