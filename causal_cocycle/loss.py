# in loss_functions_new.py, modify CocycleLoss as follows:
import torch
import torch.nn as nn

class CocycleLoss(nn.Module):
    def __init__(self, loss_callable, kernel):
        super().__init__()
        self.loss_callable = loss_callable
        self.kernel = kernel
        
    def forward(self, model, inputs, outputs):
            return self.loss_callable(model, inputs, outputs)
        
    def __call__(self, model, inputs, outputs):
            return self.forward(model, inputs, outputs)

class FlowLoss(nn.Module):
    def __init__(self, log_det=True):
        super().__init__()
        self.log_det = log_det

        def loss_callable(model, inputs, outputs):
            # The model is expected to have a .base_distribution attribute.
            if self.log_det:
                U, logdet = model.inverse_transformation(inputs, outputs)
                return torch.mean(-model.base_distribution.log_prob(U) - logdet)
            else:
                U = model.inverse_transformation(inputs, outputs)
                return torch.mean(-model.base_distribution.log_prob(U))
        
        self.loss_callable = loss_callable

    def forward(self, model, inputs, outputs):
        return self.loss_callable(model, inputs, outputs)

    def __call__(self, model, inputs, outputs):
        return self.forward(model, inputs, outputs)
