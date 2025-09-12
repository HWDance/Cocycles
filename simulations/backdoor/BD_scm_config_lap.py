"""Optimization configs"""

opt_config = {
    "epochs": 100,            # Number of epochs per optimisation run.
    "batch_size": 64,         # Training batch size.
    "val_batch_size": 1024,         # Training batch size.
    "scheduler": True,        # Whether to use a learning rate scheduler.
    "schedule_milestone": 1,  # When to call learning_rate scehdule (# epochs)
    "lr_mult": 0.9,           # Learning rate scheduler multiplier
    "learn_rate": 1e-3,       # Base learning rate for model parameters.
    "weight_decay": 1e-3,     # Weight decay factor.
}

""" Model configs"""
# Hyperparameter candidates for cross-validation.
hyper_params = {
    "weight_decay": [1e-3]
}

# Conditioner configurations: each inner list is a composite configuration.
RQS_bins = 8
conditioner_configs = [
    [
        {"type": "NN", "activation": "RELU", "output_dim": 1},
        {"type": "CONSTANT", "init": [1.0]},
        {"type": "CONSTANT", "init":  [3 for i in range(3*RQS_bins+2)]},
    ],
    [ {"type": "NN", "activation": "RELU", "output_dim": 1},
      {"type": "NN", "activation": "RELU", "output_dim": 1},
      {"type": "CONSTANT", "init":  [3 for i in range(3*RQS_bins+2)]}
    ],
    [ {"type": "NN", "activation": "RELU", "output_dim": 1},
      {"type": "NN", "activation": "RELU", "output_dim": 1},
      {"type": "NN", "activation": "RELU", "output_dim": 3 * RQS_bins + 2}
    ]
]

# Transformer configurations: each inner list is a list of transformer layer names.
# Allowed values are "Shift", "Scale", and "RQS".
transformer_configs = [
    ["Shift", "Scale", "RQS"],
    ["Shift", "Scale", "RQS"],
    ["Shift", "Scale", "RQS"]
]

base_configs = ["Laplace",
                "Laplace",
                "Laplace"
               ]

# Global model configuration dictionary.
model_config = {
    "width": 128,              # Global hidden layer width for NN conditioners.
    "layers": 2,               # Global number of layers for NN conditioners.
    "bias": True,              # Global bias flag for conditioners.
    "RQS_bins": RQS_bins,
    "conditioner_configs": conditioner_configs,
    "transformer_configs": transformer_configs,
    "base_distribution_configs": base_configs,
    "hyper_params": hyper_params
}


