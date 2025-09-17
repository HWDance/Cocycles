# Counterfactual Cocycles
Code for simulations and application in [Counterfactual Cocycles](https://arxiv.org/abs/2405.13844) (Dance and Bloem-Reddy, 2025).

# 

## Table of Contents

1. [Method at a Glance](#method-at-a-glance)
2. [Installation](#installation)  
3. [Repository Structure](#repository-structure)  
4. [Reproducing Experiments](#reproducing-experiments)
5. [Citation](#citation)
6. [License](#license)
7. [Contact](#contact)

## Method at a Glance

**Motivating Example.** For a dosage \(x\) (vs. baseline \(0\)), quantify *treatment harm* via the **dose–response harm rate**
## Method at a Glance

**Motivating Example.** For a dosage $x$ (vs. baseline $0$), quantify *treatment harm* via the **dose–response harm rate**

$$
\mathrm{THR}(x) := \mathbb{P}(Y(x)-Y(0) < 0),
$$
This requires counterfactuals for the *same unit* across dosages.

### Coherent counterfactual transports
We learn a family of transports between dosages,
$$
T_{x',x}:\ \mathcal{Y}\to\mathcal{Y},\qquad T_{x',x}\big(Y(x)\big)=Y(x'),
$$
satisfying **identity** $T_{x,x}=\mathrm{id}$ and the **cocycle** (path independence)
$$
T_{z,x}=T_{z,y}\circ T_{y,x}.
$$

### Flow-based parameterization
Each map is realized via **flows** as
$$
T_{x',x}=f_{x'}\circ f_x^{-1},
$$
where $f_x:\mathcal{Y}\to\mathbb{R}^d$ is a bijective, $x$-indexed normalizing flow (e.g., MAF/NSF with $x$ as conditioning inputs).  
This factorization **automatically** enforces the cocycle axioms:
$$
f_z\circ f_y^{-1}\circ f_y\circ f_x^{-1}=f_z\circ f_x^{-1}.
$$

### Learning (experimental or observational)
Training uses **conditional MMD (CMMD-U/V)** so that the **latent** $f_x^{-1}\!\big(Y\mid X{=}x\big)$ has the **same distribution across $x$**. This makes the latent a *shared base* **without choosing or fixing** a parametric base density.

- **Avoiding base-distribution issues:** classical “causal flows” typically assume a **specific base** (Gaussian/Laplace/$t$, etc.). Model fit and counterfactuals can be sensitive to that choice.  
  Here, the **shared base is learned implicitly** via CMMD alignment in latent space, so counterfactual transport does **not** hinge on picking the “right” base family.

**Confounding:** in observational data, augment by covariates $Z$ and learn $f_{x,z}$ so that the base is **shared conditional on $z$**; losses are conditioned on $Z$.

### Counterfactuals and harm/heterogeneity
Counterfactuals are imputed by transport:
$$
\hat Y(x') \;=\; T_{x',x}\big(Y(x)\big) \;=\; f_{x'}\!\left(f_x^{-1}\!\big(Y(x)\big)\right).
$$
Define the dose-specific effect $\hat\tau(x)=\hat Y(x)-\hat Y(0)$. Estimate **THR** (and other functionals like conditional quantiles) via **kernel-reweighted empirical functionals** (Nadaraya–Watson):
$$
\widehat{\mathrm{THR}}(x\mid z)
= \frac{\sum_i K_\lambda(z,Z_i)\,\mathbf{1}\{\hat\tau_i(x)<0\}}
       {\sum_i K_\lambda(z,Z_i)}
$$

## Installation

> Prereqs: Conda (miniconda/mambaforge) installed; Git available.

1. **Clone the repository**
   ```bash
   git clone https://github.com/HWDance/Cocycles.git
   cd Cocycles
   ```
   
2. **Create and activate the environment**
  ```bash
  conda env create -f environment.yml
  conda activate cocycles
  ```
## Repository Structure
```bash
Cocycles/
├── simulations/        # .py runners (and *_hpc.py for clusters)
├── examples/           # notebooks + figure code
├── applications/       # 401k example
├── causal_cocycle/     # core library (models, training, kernels)
├── environment.yml
├── requirements.txt
├── setup.py
└── README.md

```
## Reproducing Experiments 

## Citation
```bibtex
@article{dance2025cocycles,
  title   = {Counterfactual Cocycles: A Framework for Robust and Coherent Counterfactual Transports},
  author  = {Dance, Hugh and Bloem-Reddy, Benjamin},
  journal = {arXiv preprint arXiv:2405.13844},
  year    = {2025},
  url     = {https://arxiv.org/abs/2405.13844}
}
```

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact:

Hugh W. Dance,
PhD Researcher, Machine Learning,
Gatsby Computational Neuroscience Unit, UCL
uctphwd@ucl.ac.uk
