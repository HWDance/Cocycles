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

class CocycleMultiLoss(nn.Module):
    def __init__(self, models, losses):
        super().__init__()
        self.models = models
        self.losses = losses

    def forward(self, _, inputs, outputs):
        loss_total = 0.0
        for j, (model, loss) in enumerate(zip(self.models, self.losses)):
            x_j = inputs[:, :j+1]       # context: V_{<j+1}
            y_j = outputs[:, j:j+1]     # target:  V_{j+1}
            loss_total += loss(model, x_j, y_j)
        return loss_total

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
