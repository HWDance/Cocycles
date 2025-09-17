# Counterfactual Cocycles
Code for simulations and application in [Counterfactual Cocycles](https://arxiv.org/abs/2405.13844) (Dance and Bloem-Reddy, 2025).

### Overview
We model counterfactuals via a system of transports $\{T_{x',x}:\mathcal{Y}\to\mathcal{Y}\}_{x,x' \mathbb X}$ between treatment levels $x \in \mathbb X$, written

$$T_{x',x}:\mathcal{Y}\to\mathcal{Y}$ with $T_{x',x}(Y(x))=Y(x')$$

These maps satisfy the (cocycle) axioms of a coherent counterfactual model:
*identity* $T_{x,x}=\mathrm{id}$ and *path independence*
$T_{z,x}=T_{z,y}\circ T_{y,x}$. Enforcing these axioms addresses the path–dependence that can arise when fitting transports
\emph{pairwise}—e.g.\ with OT—without global consistency constraints.

Each transport is implemented with autoregressive normalizing flows conditioned on the treatment level, i.e.  $T_{x',x}=f_{x'}\circ f_x^{-1}$,
and trained via conditional MMD (CMMD) to match the relevant counterfactual
marginals. The resulting model behaves like an SCM with a latent noise distribution learned implicitly. Thus, no fixed base density (e.g., Gaussian/Laplace) needs to be chosen, avoiding
tail/support mis–specification sensitivity in typical causal normalizing flows. After fitting, counterfactuals are
imputed by transport, and quantities of interest (e.g., dose–response treatment harm rate and
conditional quantiles/means) are estimated with simple empirical or kernel–weighted summaries.

## Table of Contents
1. [Installation](#installation)  
3. [Repository Structure](#repository-structure)  
4. [Reproducing Experiments](#reproducing-experiments)
5. [Citation](#citation)
6. [License](#license)
7. [Contact](#contact)

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
