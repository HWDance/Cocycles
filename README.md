# Counterfactual Cocycles
Code for simulations and application in [Counterfactual Cocycles](https://arxiv.org/abs/2405.13844) (Dance and Bloem-Reddy, 2025).

We model counterfactuals via a **system of transports** between treatment levels that satisfy
identity and path-independence (cocycle), giving a **globally coherent** counterfactual model.
Each transport is implemented with an $x$-indexed **normalizing flow**, and we train flows so their
**latent representations match across treatments** via **CMMD**. This implicitly learns a **shared
noise distribution**—avoiding fragile choices of fixed base densities (e.g., Gaussian/Laplace) used
in standard causal flows. After fitting, counterfactuals are imputed by transport, and quantities
like the **dose–response treatment harm rate** and **conditional quantiles** are estimated with simple
kernel-weighted empirical summaries.

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
