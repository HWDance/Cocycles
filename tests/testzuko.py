import time
import torch
import torch.nn as nn
from torch.distributions import Normal

# Imports for your modules
from testarchitectures import get_stock_transforms
from causal_cocycle.model_new import ZukoCocycleModel, ZukoFlowModel

# Fix random seed for reproducibility
torch.manual_seed(0)

# Dummy data
N, x_dim, y_dim = 100, 1, 2
X = torch.randn(N, x_dim)
Y = torch.randn(N, y_dim)

# Pick a test transform
transforms = get_stock_transforms(x_dim=x_dim, y_dim=y_dim)
test_transform = transforms[0]

print("\n=== Timing ZukoCocycleModel ===")
cocycle_model = ZukoCocycleModel(nn.ModuleList([test_transform]))

# 1) inverse_transformation
start = time.perf_counter()
U = cocycle_model.inverse_transformation(X, Y)
t_inv = time.perf_counter() - start
print(f"inverse_transformation: {t_inv*1000:.2f} ms")

# 2) transformation
start = time.perf_counter()
Y_rec = cocycle_model.transformation(X, U)
t_fwd = time.perf_counter() - start
print(f"transformation: {t_fwd*1000:.2f} ms")

# 3) cocycle
start = time.perf_counter()
v = cocycle_model.cocycle(X, X, Y)
t_cocycle = time.perf_counter() - start
print(f"cocycle: {t_cocycle*1000:.2f} ms, output shape: {v.shape}")

# 4) cocycle_outer
start = time.perf_counter()
outer = cocycle_model.cocycle_outer(X, X, Y)
t_outer = time.perf_counter() - start
print(f"cocycle_outer: {t_outer*1000:.2f} ms, output shape: {outer.shape}")

print("\n=== Timing ZukoFlowModel ===")
base_dist = Normal(torch.zeros(y_dim), torch.ones(y_dim))
flow_model = ZukoFlowModel(nn.ModuleList([test_transform]), base_dist)

# 1) inverse_transformation
start = time.perf_counter()
U_flow = flow_model.inverse_transformation(X, Y)
t_inv_f = time.perf_counter() - start
print(f"inverse_transformation: {t_inv_f*1000:.2f} ms")

# 2) transformation
start = time.perf_counter()
Y_flow_rec = flow_model.transformation(X, U_flow)
t_fwd_f = time.perf_counter() - start
print(f"transformation: {t_fwd_f*1000:.2f} ms")

# 3) log_prob
start = time.perf_counter()
lp = flow_model.log_prob(X, Y)
t_logp = time.perf_counter() - start
print(f"log_prob: {t_logp*1000:.2f} ms, output shape: {lp.shape}")

# 4) sample
start = time.perf_counter()
samples = flow_model.sample(X, num_samples=5)
t_sample = time.perf_counter() - start
print(f"sample: {t_sample*1000:.2f} ms, output shape: {samples.shape}")

# 5) flow cocycle
start = time.perf_counter()
fv = flow_model.cocycle(X, X, Y)
t_cocycle_f = time.perf_counter() - start
print(f"cocycle: {t_cocycle_f*1000:.2f} ms, output shape: {fv.shape}")

# 6) flow cocycle_outer
start = time.perf_counter()
f_outer = flow_model.cocycle_outer(X, X, Y)
t_outer_f = time.perf_counter() - start
print(f"cocycle_outer: {t_outer_f*1000:.2f} ms, output shape: {f_outer.shape}")

