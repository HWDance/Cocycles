import torch
import torch.nn as nn

class Functional(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, Ytrain, Xtrain, Xtest):
        raise NotImplementedError

    def get_weights(self, Ytrain, Xtrain):
        raise NotImplementedError

    def get_features(self, Xtrain):
        raise NotImplementedError


class NWFunctional(Functional):
    def __init__(self, kernel, reg=0.0):
        super().__init__(kernel)
        self.reg = reg
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0))  # log-scale for stability

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)

    def forward(self, Ytrain, Xtrain, Xtest):
        K_xtestx = self.kernel.get_gram(Xtest, Xtrain) + self.reg
        numer = K_xtestx @ Ytrain
        denom = K_xtestx.sum(dim=1, keepdim=True) + 1e-8  # avoid division by zero
        return numer / denom

class LLFunctional(Functional):
    def __init__(self, kernel, reg=1e-6):
        super().__init__(kernel)
        self.reg = reg
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0))  # Learn in log-space

    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)

    def forward(self, Ytrain, Xtrain, Xtest):
        device = Xtrain.device
        K = self.kernel.get_gram(Xtest, Xtrain)  # Shape: (n_test, n_train)

        n_test = Xtest.shape[0]
        ypred = torch.zeros((n_test, 1), device=device)

        ones = torch.ones((Xtrain.shape[0], 1), device=device)
        Xtild = torch.cat((ones, Xtrain), dim=1)  # (n_train, d+1)

        for i in range(n_test):
            K_i = K[i].unsqueeze(1)  # (n_train, 1)
            W = K_i * Xtild  # Weight each row
            XWX = Xtild.T @ W  # (d+1, d+1)
            XWy = (Xtild.T @ (Ytrain * K_i)).squeeze()  # (d+1,)

            reg_mat = self.reg * torch.eye(XWX.shape[0], device=device)
            beta = torch.linalg.solve(XWX + reg_mat, XWy)
            x_test_tild = torch.cat((torch.tensor([1.0], device=device), Xtest[i])).unsqueeze(0)
            ypred[i] = x_test_tild @ beta.unsqueeze(1)

        return ypred

class KRRFunctional(Functional):
    def __init__(self, kernel, penalty=1e-2, reg=1e-6):
        super().__init__(kernel)
        self.reg = reg
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0))
        self.log_penalty = nn.Parameter(torch.tensor(np.log(penalty)))
        
    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)
    
    @property
    def penalty(self):
        return torch.exp(self.log_penalty)

    def forward(self, Ytrain, Xtrain, Xtest):
        K_xx = self.kernel.get_gram(Xtrain, Xtrain)
        K_xXtest = self.kernel.get_gram(Xtest, Xtrain)
        n = Ytrain.shape[0]

        ridge_term = (self.penalty + self.reg) * torch.eye(n, device=K_xx.device)
        alpha = torch.linalg.solve(K_xx + ridge_term, Ytrain)
        return K_xXtest @ alpha
