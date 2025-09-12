"""Linear Model Optimization configs"""

opt_config = {
    "epochs": 500,            # Number of epochs per optimisation run.
    "batch_size": 128,         # Training batch size.
    "val_batch_size": 1024,         # Training batch size.
    "scheduler": False,        # Whether to use a learning rate scheduler.
    "learn_rate": 1e-2,       # Base learning rate for model parameters.
}

""" Model configs"""
# Hyperparameter candidates for cross-validation.
hyper_params = {
    "weight_decay": [0]
}

# Conditioner configurations: each inner list is a composite configuration.
# Each dictionary now only needs "type", "activation", and "output_dim".
RQS_bins = 8
conditioner_configs = [
    [ {"type": "LINEAR", "output_dim": 1} ],
    [ {"type": "NN", "activation": "RELU", "output_dim": 1} ],
    [ {"type": "NN", "activation": "RELU", "output_dim": 1},
      {"type": "NN", "activation": "RELU", "output_dim": 1} ],
    [ {"type": "NN", "activation": "RELU", "output_dim": 1},
      {"type": "NN", "activation": "RELU", "output_dim": 1},
      {"type": "NN", "activation": "RELU", "output_dim": 3 * RQS_bins + 2} ]
]

# Transformer configurations: each inner list is a list of transformer layer names.
# Allowed values are "Shift", "Scale", and "RQS".
transformer_configs = [
    ["Shift"],
    ["Shift"],
    ["Shift", "Scale"],
    ["Shift", "Scale", "RQS"]
]

# Global model configuration dictionary.
model_config = {
    "width": 64,              # Global hidden layer width for NN conditioners.
    "layers": 2,               # Global number of layers for NN conditioners.
    "bias": True,              # Global bias flag for conditioners.
    "RQS_bins": RQS_bins,
    "conditioner_configs": conditioner_configs,
    "transformer_configs": transformer_configs,
    "hyper_params": hyper_params
}


