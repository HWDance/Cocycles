# model_builder.py
import torch
import torch.nn as nn
from torch.distributions import Normal, Laplace, StudentT
from causal_cocycle.base_distributions import ParameterizedNormal, ParameterizedLaplace, ParameterizedStudentT
from causal_cocycle.model_new import CocycleModel, FlowModel 
from causal_cocycle.conditioners_new import CompositeConditioner, NNConditioner, LinConditioner, ConstantConditioner
from causal_cocycle.transformers_new import Transformer, ShiftLayer, ScaleLayer, RQSLayer

class CocycleFactory:
    def __init__(self, input_dim, config):
        """
        Initialize the factory with a global input dimension and additional configuration.
        
        Parameters:
        -----------
        input_dim : int
            Global dimensionality of the input X. This is determined at the dataset/task level.
        config : dict
            A configuration dictionary with the following optional keys:
              - "width": int, default 128, hidden layer width for NN conditioners.
              - "layers": int, default 2, number of layers for NN conditioners.
              - "bias": bool, default True, whether to include a bias in the conditioners.
              - "RQS_bins": int, default 8, number of bins for an RQS layer.
              - "conditioner_configs": list of lists (each inner list is a list of dictionaries),
                   where each dictionary specifies a single conditioner. Each dictionary should include:
                      "type": string, either "NN" or "Linear".
                      For "NN", an "activation" key (e.g. "RELU", "ELU", "TANH") and "output_dim" (default 1).
                   If not provided, a default configuration is used.
              - "transformer_configs": list of lists, where each inner list is a list of strings
                   indicating transformer layers to use (allowed: "Shift", "Scale", "RQS").
                   If not provided, a default configuration is used.
              - "hyper_params": dict, optional, mapping hyperparameter names to candidate values,
                   e.g. {"weight_decay": [1e-3, 1e-4]}.
        """
        self.input_dim = input_dim
        self.width = config.get("width", 128)
        self.layers = config.get("layers", 2)
        self.bias = config.get("bias", True)
        self.RQS_bins = config.get("RQS_bins", 8)
        
        # Global input dimension for conditioners.
        # For instance, if you want to include an intercept or extra feature, you could do input_dim + 1.
        # Here, we'll simply use input_dim.
        self.global_input_dim = self.input_dim
        
        self.conditioner_configs = config.get("conditioner_configs", None)
        self.transformer_configs = config.get("transformer_configs", None)
        self.hyper_params = config.get("hyper_params", {})
        
        # Set default conditioner and transformer configurations if not provided.
        if self.conditioner_configs is None:
            self.conditioner_configs = [
                [ {"type": "NN", "activation": "RELU", "output_dim": 1} ]
            ]
        if self.transformer_configs is None:
            self.transformer_configs = [
                ["Shift"]
            ]
    
    def build_models(self):
        """
        Constructs candidate CocycleModel instances according to the configuration.
        
        Returns:
          - models: list of CocycleModel instances.
          - hyper_args: list of dictionaries containing hyperparameter settings for each candidate.
        """
        models = []
        hyper_args = []
        
        # Assume one-to-one mapping between conditioner_configs and transformer_configs.
        num_configs = min(len(self.conditioner_configs), len(self.transformer_configs))
        for idx in range(num_configs):
            cond_config = self.conditioner_configs[idx]
            trans_config = self.transformer_configs[idx]
            
            # Build individual conditioner modules using only local settings.
            cond_modules = []
            for spec in cond_config:
                cond_type = spec.get("type", "NN").upper()
                if cond_type == "NN":
                    # Determine activation based on the string.
                    act_str = spec.get("activation", "RELU").upper()
                    if act_str == "RELU":
                        activation = nn.ReLU()
                    elif act_str == "ELU":
                        activation = nn.ELU()
                    elif act_str == "TANH":
                        activation = nn.Tanh()
                    else:
                        raise ValueError(f"Unsupported activation: {act_str}")
                    # Create NNConditioner using global parameters and local output_dim.
                    cond = NNConditioner(width=self.width,
                                         layers=self.layers,
                                         input_dims=self.global_input_dim,
                                         output_dims=spec.get("output_dim", 1),
                                         bias=self.bias,
                                         activation=activation)
                    cond_modules.append(cond)
                elif cond_type == "LINEAR":
                    cond = LinConditioner(d=self.global_input_dim,
                                          p=spec.get("output_dim", 1),
                                          bias=self.bias)
                    cond_modules.append(cond)
                else:
                    raise ValueError(f"Unsupported conditioner type: {cond_type}")
            
            # Wrap individual conditioners in a CompositeConditioner.
            composite_cond = CompositeConditioner(cond_modules)
            
            # Build transformer layers.
            trans_layers = []
            for layer in trans_config:
                layer = layer.upper()
                if layer == "SHIFT":
                    trans_layers.append(ShiftLayer())
                elif layer == "SCALE":
                    trans_layers.append(ScaleLayer())
                elif layer == "RQS":
                    trans_layers.append(RQSLayer(self.RQS_bins))
                else:
                    raise ValueError(f"Unsupported transformer layer: {layer}")
            transformer = Transformer(trans_layers, logdet=False)
            
            # Create the base CocycleModel.
            base_model = CocycleModel(composite_cond, transformer)
            
            # If hyperparameters are provided, create candidate models for each candidate value.
            if self.hyper_params:
                # For simplicity, assume one hyperparameter key.
                for hyper_name, candidate_list in self.hyper_params.items():
                    for candidate in candidate_list:
                        models.append(CocycleModel(composite_cond, transformer))
                        hyper_args.append({hyper_name: candidate})
            else:
                models.append(base_model)
                hyper_args.append({})
        
        return models, hyper_args

class FlowFactory:
    def __init__(self, input_dim, config):
        """
        Initialize the factory with global input and output dimensions, a configuration dictionary,
        and an optional base distribution.
        
        Parameters:
        -----------
        input_dim : int
            Global dimensionality of the input X.
        output_dim : int
            Global dimensionality of the output Y.
        config : dict
            A configuration dictionary with optional keys:
              - "width": int, default 128, hidden layer width for NN conditioners.
              - "layers": int, default 2, number of layers for NN conditioners.
              - "bias": bool, default True, whether to include a bias in the conditioners.
              - "RQS_bins": int, default 8, number of bins for an RQS layer.
              - "conditioner_configs": list of lists, where each inner list is a list of dictionaries.
                   Each dictionary should include:
                      "type": string, either "NN" or "Linear".
                      For "NN", an "activation" key (e.g. "RELU", "ELU", "TANH") and "output_dim" (default 1).
                   If not provided, a default configuration is used.
              - "transformer_configs": list of lists, where each inner list is a list of transformer layer names.
                   Allowed names are "Shift", "Scale", and "RQS".
                   If not provided, a default configuration is used.
              - "hyper_params": dict, optional, mapping hyperparameter names to candidate values.
        base_distribution : torch.distributions.Distribution, optional
            The base distribution for the flow model. If not provided, defaults to Normal(0, 1)
            on the output dimension.
        """
        self.input_dim = input_dim
        self.width = config.get("width", 128)
        self.layers = config.get("layers", 2)
        self.bias = config.get("bias", True)
        self.RQS_bins = config.get("RQS_bins", 8)
        
        
        # Global input dimension for conditioners.
        self.global_input_dim = self.input_dim
        
        self.conditioner_configs = config.get("conditioner_configs", None)
        self.transformer_configs = config.get("transformer_configs", None)
        self.base_distribution_configs = config.get("base_distribution_configs", None)
        self.hyper_params = config.get("hyper_params", {})
        
        if self.conditioner_configs is None:
            self.conditioner_configs = [[{"type": "NN", "activation": "RELU", "output_dim": 1}]]
        if self.transformer_configs is None:
            self.transformer_configs = [["Shift"]]
        if self.base_distribution_configs is None:
            self.base_distribution_configs = ["Normal"]

    def build_models(self):
        """
        Constructs candidate FlowModel instances according to the configuration.
        Each FlowModel will have an attribute .base_distribution.
        
        Returns:
        --------
        models : list of FlowModel instances
            The list of candidate flow-based models.
        hyper_args : list of dict
            A list of hyperparameter dictionaries corresponding to each candidate.
        """
        models = []
        hyper_args = []
        num_configs = min(len(self.conditioner_configs), len(self.transformer_configs))
        for idx in range(num_configs):
            cond_config = self.conditioner_configs[idx]
            trans_config = self.transformer_configs[idx]
            base_config = self.base_distribution_configs[idx]
            
            # Build individual conditioner modules.
            cond_modules = []
            for spec in cond_config:
                cond_type = spec.get("type", "NN").upper()
                if cond_type == "NN":
                    act_str = spec.get("activation", "RELU").upper()
                    if act_str == "RELU":
                        activation = nn.ReLU()
                    elif act_str == "ELU":
                        activation = nn.ELU()
                    elif act_str == "TANH":
                        activation = nn.Tanh()
                    else:
                        raise ValueError(f"Unsupported activation: {act_str}")
                    local_out = spec.get("output_dim", 1)
                    cond = NNConditioner(width=self.width,
                                         layers=self.layers,
                                         input_dims=self.global_input_dim,
                                         output_dims=local_out,
                                         bias=self.bias,
                                         activation=activation)
                    cond_modules.append(cond)
                elif cond_type == "LINEAR":
                    local_out = spec.get("output_dim", 1)
                    cond = LinConditioner(d=self.global_input_dim,
                                          p=local_out,
                                          bias=self.bias)
                    cond_modules.append(cond)
                elif cond_type == "CONSTANT":
                    # Use the constant conditioner with a default constant value (or read from spec).
                    # You can also allow grad to be True or False via the config.
                    init = spec.get("init", 1.0)
                    full = spec.get("full", True)
                    grad = spec.get("grad", True)
                    cond = ConstantConditioner(init=init, full=full, grad=grad)
                    cond_modules.append(cond)
                else:
                    raise ValueError(f"Unsupported conditioner type: {cond_type}")
            composite_cond = CompositeConditioner(cond_modules)
            
            # Build transformer layers.
            trans_layers = []
            for layer in trans_config:
                layer = layer.upper()
                if layer == "SHIFT":
                    trans_layers.append(ShiftLayer())
                elif layer == "SCALE":
                    trans_layers.append(ScaleLayer())
                elif layer == "RQS":
                    trans_layers.append(RQSLayer(self.RQS_bins))
                else:
                    raise ValueError(f"Unsupported transformer layer: {layer}")
            transformer = Transformer(trans_layers, logdet=True)

            # Build base distributions
            if base_config == "Normal":
                base = ParameterizedNormal()
            if base_config == "Laplace":
                base = ParameterizedLaplace()
            if base_config == "StudentT":
                base = ParameterizedStudentT()
            
            base_model = FlowModel(composite_cond, transformer, base)
            
            if self.hyper_params:
                for hyper_name, candidate_list in self.hyper_params.items():
                    for candidate in candidate_list:
                        model_instance = FlowModel(composite_cond, transformer, base)
                        models.append(model_instance)
                        hyper_args.append({hyper_name: candidate})
            else:
                models.append(base_model)
                hyper_args.append({})
        return models, hyper_args


# Example usage:
if __name__ == "__main__":

    # Global input_dim is determined from the DGP configuration (e.g., from dgp_config.py).
    global_input_dim = 2  # for example, treatment concatenated with confounders.
    config = {
        "width": 128,
        "layers": 2,
        "bias": True,
        "RQS_bins": 8,
        "conditioner_configs": [
            [ {"type": "NN", "activation": "RELU", "output_dim": 1} ],
            [ {"type": "NN", "activation": "RELU", "output_dim": 1},
              {"type": "NN", "activation": "RELU", "output_dim": 1} ]
        ],
        "transformer_configs": [
            ["Shift"],
            ["Shift", "Scale"]
        ],
        "hyper_params": {"weight_decay": [1e-3, 1e-4]}
    }
    
    factory = CocycleFactory(global_input_dim, config)
    models, hyper_args = factory.build_models()
    print(f"Built {len(models)} candidate models with hyperparameters: {hyper_args}")

    base_configs = ["Normal"]
    
    conditioner_configs = [
        [{"type": "NN", "activation": "RELU", "output_dim": 1}],
        [{"type": "NN", "activation": "RELU", "output_dim": 1},
         {"type": "NN", "activation": "RELU", "output_dim": 1}]
    ]
    transformer_configs = [
        ["Shift"],
        ["Shift", "Scale"]
    ]
    model_config = {
        "width": 128,
        "layers": 2,
        "bias": True,
        "RQS_bins": 7,
        "conditioner_configs": conditioner_configs,
        "transformer_configs": transformer_configs,
        "base_distribution_configs": base_configs,
        "hyper_params": hyper_params
    }
    
    factory = FlowFactory(global_input_dim, global_output_dim, model_config)
    models, hyper_args = factory.build_models()
    print(f"Built {len(models)} candidate models with hyperparameters: {hyper_args}")



