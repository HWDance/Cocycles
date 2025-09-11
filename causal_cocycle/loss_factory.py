# loss_factory.py
import torch
from torch.distributions import Normal
from causal_cocycle.loss import CocycleLoss

class CocycleLossFactory:
    def __init__(self, kernel):
        """
        Initialize the loss factory.
        
        Parameters:
        -----------
        kernel : list
            A list of kernel objects. Each kernel must have a 'lengthscale' attribute
            and a 'get_gram' method.
            An optional parameter for subsampling if needed.
        """
        self.kernel = kernel
        # Map loss type strings to internal methods.
        self.loss_mapping = {
            "HSIC": self._hsic,
            "CMMD_V": self._cmmd_v,
            "CMMD_U": self._cmmd_u,
            "URR": self._urr,
            "URR_N": self._urr_n,
            "LS": self._ls
        }
    
    def _set_median_heuristic(self, X, Y, subsamples=10000):
        """
        Compute and set the median-heuristic lengthscale for the kernels using the data.
        For the first kernel (assumed for inputs) and, if available, for the second kernel (for outputs).
        """
        # Subsample if needed.
        if subsamples < len(X):
            indices = torch.randperm(len(X))[:subsamples]
            X_sub = X[indices]
            Y_sub = Y[indices] if Y is not None else None
        else:
            X_sub = X
            Y_sub = Y
        
        # For the first kernel.
        Dist = torch.cdist(X_sub, X_sub, p=2.0) ** 2
        lower_tri = torch.tril(Dist, diagonal=-1)
        mask = torch.tril(torch.ones_like(lower_tri, dtype=torch.bool), -1)
        self.kernel[0].lengthscale = (lower_tri[mask].median() / 2).sqrt()
        
        # If a second kernel exists and Y is provided, set its lengthscale.
        if Y is not None and len(self.kernel) > 1:
            Dist = torch.cdist(Y_sub, Y_sub, p=2.0) ** 2
            lower_tri = torch.tril(Dist, diagonal=-1)
            mask = torch.tril(torch.ones_like(lower_tri, dtype=torch.bool), -1)
            self.kernel[1].lengthscale = (lower_tri[mask].median() / 2).sqrt()
            
    def _hsic(self, model, inputs, outputs):
        """
        Compute the HSIC loss.
        """
        U = model.inverse_transformation(inputs, outputs)
        m = len(outputs)
        K_xx = self.kernel[0].get_gram(inputs, inputs)
        K_uu = self.kernel[1].get_gram(U, U)
        H = torch.eye(m, device=inputs.device) - (1.0/m)*torch.ones((m, m), device=inputs.device)
        return torch.trace(K_xx @ H @ K_uu @ H) / (m**2)
    
    def _cmmd_v(self, model, inputs, outputs):
        """
        Compute the V-statistic version of CMMD.
        """
        n = len(outputs)
        outputs_pred = model.cocycle_outer(inputs, inputs, outputs)
        if outputs_pred.dim() < 3:
            outputs_pred = outputs_pred[..., None]
        K = 0.0
        batchsize = n
        if n**3 >= 1e8:
            batchsize = max(1, min(n, int(1e8 / n**2)))
        nbatch = int(n / batchsize)
        for i in range(nbatch):
            K += (self.kernel[1].get_gram(outputs_pred[i*batchsize:(i+1)*batchsize],
                                            outputs_pred[i*batchsize:(i+1)*batchsize]).sum() / (n**3))
            K += (-2 * self.kernel[1].get_gram(outputs[i*batchsize:(i+1)*batchsize, None, :],
                                                outputs_pred[i*batchsize:(i+1)*batchsize]).sum() / (n**2))
        return K
    
    def _cmmd_u(self, model, inputs, outputs):
        """
        Compute the U-statistic version of CMMD.
        """
        n = len(outputs)
        Mask = torch.ones((n, n, n), device=outputs.device)
        for i in range(n):
            Mask[i, :, i] = 0
            Mask[i, i, :] = 0
            Mask[:, i, i] = 0
        outputs_pred = model.cocycle_outer(inputs, inputs, outputs)
        if outputs_pred.dim() < 3:
            outputs_pred = outputs_pred[..., None]
        K1 = self.kernel[1].get_gram(outputs_pred, outputs_pred)
        K2 = -2 * self.kernel[1].get_gram(outputs_pred, outputs[:, None, :])[..., 0]
        return (K1 * Mask).sum() / (n*(n-1)*(n-2)) + (K2 * (1 - torch.eye(n, device=outputs.device))).sum() / (n*(n-1))
    
    def _urr(self, model, inputs, outputs):
        """
        URR loss with a single draw per sample (for training with rsample)
        """
        Dist = model.base_distribution # Normal(0, 1)
        output_samples_1 = model.transformation(inputs, Dist.rsample((len(inputs), 1)))
        output_samples_2 = model.transformation(inputs, Dist.rsample((len(inputs), 1)))
        K1 = self.kernel[1].get_gram(output_samples_1[..., None], output_samples_2[..., None])
        K2 = self.kernel[1].get_gram(output_samples_1[..., None], outputs[..., None])
        return (K1 - 2 * K2).mean()

    def _urr_n(self, model, inputs, outputs):
        """
        URR loss with Monte‑Carlo using n independent draws. (for eval)
        """
        n = inputs.size(0)
        Dist = model.base_distribution   # e.g. Normal(0,1)
        K = self.kernel[1].get_gram     # shorthand

        def one_draw():
            # sample two independent noise vectors of shape (n,1)
            z1 = Dist.sample((n,1))
            z2 = Dist.sample((n,1))
            # push through flow
            y1 = model.transformation(inputs, z1)
            y2 = model.transformation(inputs, z2)
            # compute full‐Gram averages
            return ( K(y1[...,None], y2[...,None])
                   - 2*K(y1[...,None], outputs[...,None]) ).mean()

        # repeat n times and average
        vals = torch.stack([one_draw() for _ in range(n)])
        return vals.mean()
    
    def _ls(self, model, inputs, outputs):
        """
        Compute the least squares loss
        """
        return (model.inverse_transformation(inputs, outputs).abs() ** 2).mean()
    
    def build_loss(self, loss_type, X, Y, subsamples=10000):
        """
        Build and return a CocycleLoss object configured with the chosen loss function
        and kernels with median-heuristic tuned hyperparameters.
        
        Parameters:
        -----------
        loss_type : str
            A string indicating which loss function to use (e.g. "HSIC", "CMMD_V", "CMMD_U", "URR", "LS").
        X : torch.Tensor
            Input data (N x d) for tuning kernel hyperparameters.
        Y : torch.Tensor
            Output data (N x p) for tuning kernel hyperparameters.
        subsamples : int, default 10000
            Number of samples to use for computing the median heuristic.
        
        Returns:
        --------
        final_loss : CocycleLoss
            A loss object that, when called with (model, X, Y), returns the computed loss.
        """
        if loss_type not in self.loss_mapping:
            raise ValueError(f"Unknown loss type '{loss_type}'. Supported types: {list(self.loss_mapping.keys())}")
        # Tune the kernel hyperparameters using the median heuristic.
        self._set_median_heuristic(X, Y, subsamples)
        
        # Select the callable loss function.
        loss_callable = self.loss_mapping[loss_type]
        
        # Instantiate and return a final CocycleLoss object that holds the kernel and callable.
        return CocycleLoss(loss_callable, self.kernel)

