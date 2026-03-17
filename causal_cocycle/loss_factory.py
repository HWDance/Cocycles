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
            "WCMMD_V": self._wcmmd_v,
            "CMMD_U": self._cmmd_u,
            "URR": self._urr,
            "URR_N": self._urr_n,
            "LS": self._ls
        }

    def _num_samples(self, X):
        if isinstance(X, dict):
            for key, value in X.items():
                if key == "__idx__":
                    continue
                return value.shape[0]
            raise ValueError("Input dict must contain at least one non-index tensor.")
        return X.shape[0]

    def _kernel_input_tensor(self, X):
        if not isinstance(X, dict):
            return X

        ordered_keys = []
        if "X" in X:
            ordered_keys.append("X")
        if "Z" in X:
            ordered_keys.append("Z")
        ordered_keys.extend(sorted(k for k in X.keys() if k not in {"X", "Z", "__idx__"}))

        if not ordered_keys:
            raise ValueError("Input dict must contain at least one tensor for kernel computations.")

        pieces = []
        for key in ordered_keys:
            value = X[key]
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            pieces.append(value)
        return torch.cat(pieces, dim=-1)
    
    def _set_median_heuristic(self, X, Y, subsamples=10000):
        """
        Compute and set the median-heuristic lengthscale for the kernels using the data.
        For the first kernel (assumed for inputs) and, if available, for the second kernel (for outputs).
        """
        X_tensor = self._kernel_input_tensor(X)

        # Subsample if needed.
        if subsamples < self._num_samples(X):
            indices = torch.randperm(self._num_samples(X))[:subsamples]
            X_sub = X_tensor[indices]
            Y_sub = Y[indices] if Y is not None else None
        else:
            X_sub = X_tensor
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

    def _resolve_weights(
        self,
        inputs,
        outputs,
        weights=None,
        weight_fn=None,
        normalize=False,
        require_full_batch=False,
    ):
        n = len(outputs)

        if (weights is None) == (weight_fn is None):
            raise ValueError("Provide exactly one of `weights` or `weight_fn` for WCMMD_V.")

        if weight_fn is not None:
            W = weight_fn(inputs)
        else:
            W = weights

        if not torch.is_tensor(W):
            W = torch.as_tensor(W, device=outputs.device, dtype=outputs.dtype)
        else:
            W = W.to(device=outputs.device, dtype=outputs.dtype)

        if W.shape != (n, n):
            if require_full_batch:
                raise ValueError(
                    "Fixed WCMMD_V weights were precomputed for the full support set with shape "
                    f"{tuple(W.shape)}, but the current loss call received a batch of size {n}. "
                    "Fixed mode currently assumes full-batch loss evaluation."
                )
            raise ValueError(f"Weights must have shape {(n, n)}, got {tuple(W.shape)}.")

        if normalize:
            W = W / W.sum(dim=1, keepdim=True).clamp_min(torch.finfo(W.dtype).eps)

        return W

    def _wcmmd_v(
        self,
        model,
        inputs,
        outputs,
        weights=None,
        weight_fn=None,
        normalize_weights=False,
        require_full_batch=False,
    ):
        """
        Compute the weighted V-statistic CMMD loss

            -(2/n) sum_{i,j} w_ij k(Y_i, Y_T^{(i,j)})
            +(1/n) sum_{i,j,k} w_ij w_ik k(Y_T^{(i,j)}, Y_T^{(i,k)}).

        The weighting is supplied either by a fixed matrix `weights` or by
        `weight_fn(inputs)`, which should return an (n, n) tensor for the
        current batch.
        """
        n = len(outputs)
        W = self._resolve_weights(
            inputs,
            outputs,
            weights=weights,
            weight_fn=weight_fn,
            normalize=normalize_weights,
            require_full_batch=require_full_batch,
        )

        outputs_pred = model.cocycle_outer(inputs, inputs, outputs)
        if outputs_pred.dim() < 3:
            outputs_pred = outputs_pred[..., None]

        K_obs = self.kernel[1].get_gram(outputs[:, None, :], outputs_pred).squeeze(1)
        K_pred = self.kernel[1].get_gram(outputs_pred, outputs_pred)

        term_obs = -2.0 * (W * K_obs).sum() / n
        term_pred = torch.einsum('ij,ijk,ik->', W, K_pred, W) / n
        return term_obs + term_pred
    
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
    
    def build_loss(
        self,
        loss_type,
        X,
        Y,
        subsamples=10000,
        weights=None,
        weight_fn=None,
        weight_estimator=None,
        weight_mode="fixed",
        normalize_weights=False,
    ):
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

        if loss_type == "WCMMD_V":
            fixed_full_weights = False

            if weight_estimator is not None:
                if weights is not None or weight_fn is not None:
                    raise ValueError(
                        "Provide `weight_estimator` or explicit weights, not multiple weight sources."
                    )
                if weight_mode != "fixed":
                    raise ValueError(
                        f"Unsupported weight_mode '{weight_mode}' for WCMMD_V. "
                        "Only 'fixed' is currently implemented."
                    )
                weights = weight_estimator.training_weights()
                fixed_full_weights = True

            if weights is None and weight_fn is None:
                raise ValueError(
                    "WCMMD_V requires one of `weights`, `weight_fn`, or `weight_estimator`."
                )

            loss_callable = lambda model, inputs, outputs: self._wcmmd_v(
                model,
                inputs,
                outputs,
                weights=weights,
                weight_fn=weight_fn,
                normalize_weights=normalize_weights,
                require_full_batch=fixed_full_weights,
            )
        else:
            # Select the callable loss function.
            loss_callable = self.loss_mapping[loss_type]
        
        # Instantiate and return a final CocycleLoss object that holds the kernel and callable.
        return CocycleLoss(loss_callable, self.kernel)
