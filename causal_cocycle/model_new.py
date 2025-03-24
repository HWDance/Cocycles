import torch
import torch.nn as nn

class CocycleOutcomeModel(nn.Module):
    def __init__(self, model, inputs_train, outputs_train):
        """
        model: cocycle model (an nn.Module)
        inputs_train: training inputs (N x D torch.Tensor)
        outputs_train: training outputs (N x P torch.Tensor)
        """
        super().__init__()
        self.model = model
        self.inputs = inputs_train
        self.outputs = outputs_train

    def forward(self, inputs, feature):
        prediction = self.model.cocycle_outer(inputs, self.inputs, self.outputs)  # M x N x P
        return feature(prediction).mean(2)

class FlowOutcomeModel(nn.Module):
    def __init__(self, model, noise_samples):
        """
        model: flow model (an nn.Module)
        noise_samples: noise samples from base distribution (N x P torch.Tensor)
        """
        super().__init__()
        self.model = model
        self.noise = noise_samples

    def forward(self, inputs, feature):
        prediction = self.model.transformation_outer(inputs, self.noise)  # M x N x P
        return feature(prediction).mean(2)

class CocycleModel(nn.Module):
    def __init__(self, conditioner, transformer):
        """
        conditioner: a composite conditioner module that returns a list of outputs
                     (an instance of Compositeconditioner)
        transformer: an invertible transformer module
        """
        super().__init__()
        # Expect conditioner to be a composite module.
        self.conditioner = conditioner
        self.transformer = transformer

    def transformation(self, x, y):
        transformer_parameters = self.conditioner(x)  # Returns a list of outputs
        return self.transformer.forward(transformer_parameters, y)

    def transformation_outer(self, x, y):
        transformer_parameters = []
        for cond in self.conditioner.layers:  # iterate over the ModuleList
            eye_y = torch.ones((len(y), 1), device=y.device)
            eye_x = torch.ones((len(x), 1), device=x.device)
            outer_parameters = torch.kron(cond(x), eye_y)
            outer_y = torch.kron(eye_x, y)
            transformer_parameters.append(outer_parameters)
        return self.transformer.forward(transformer_parameters, outer_y).reshape(len(x), len(y))

    def inverse_transformation(self, x, y):
        transformer_parameters = self.conditioner(x)
        return self.transformer.backward(transformer_parameters, y)

    def cocycle(self, x1, x2, y):
        return self.transformation(x1, self.inverse_transformation(x2, y))

    def cocycle_outer(self, x1, x2, y):
        return self.transformation_outer(x1, self.inverse_transformation(x2, y))

class FlowModel(nn.Module):
    """
    Similar to CocycleModel but structured to support log-determinants.
    """
    def __init__(self, conditioner, transformer, base_dist):
        super().__init__()
        self.conditioner = conditioner
        self.transformer = transformer
        self.base_distribution_module = base_dist  # Register as an attribute

    @property
    def base_distribution(self):
        # When you need the distribution, call forward on the module.
        return self.base_distribution_module()

    def transformation(self, x, y):
        transformer_parameters = self.conditioner(x)
        return self.transformer.forward(transformer_parameters, y)

    def transformation_outer(self, x, y):
        transformer_parameters = []
        for cond in self.conditioner.layers:
            eye = torch.ones((len(y), 1), device=y.device)
            outer_parameters = torch.kron(cond(x), eye)
            outer_y = torch.kron(eye, y)
            transformer_parameters.append(outer_parameters)
        return self.transformer.forward(transformer_parameters, outer_y).reshape(len(x), len(y))

    def inverse_transformation(self, x, y):
        transformer_parameters = self.conditioner(x)
        return self.transformer.backward(transformer_parameters, y)

    def cocycle(self, x1, x2, y):
        return self.transformation(x1, self.inverse_transformation(x2, y))

    def cocycle_outer(self, x1, x2, y):
        return self.transformation_outer(x1, self.inverse_transformation(x2, y))