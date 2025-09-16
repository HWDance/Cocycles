"""Optimization configs"""

opt_config = {
    "epochs": 1000,            # Number of epochs per optimisation run.
    "batch_size": 128,         # Training batch size.
    "val_batch_size": 1024,         # Training batch size.
    "scheduler": True,        # Whether to use a learning rate scheduler.
    "schedule_milestone": 1,  # When to call learning_rate scehdule (# epochs)
    "lr_mult": 0.9,           # Learning rate scheduler multiplier
    "learn_rate": 1e-3,       # Base learning rate for model parameters.
    "weight_decay": 1e-3,     # Weight decay factor.
}



