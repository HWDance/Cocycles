# Counterfactual Cocycles
This is the repository to reproduce the results of [Counterfactual Cocycles](https://arxiv.org/abs/2405.13844) (Dance and Bloem-Reddy, 2025). It includes the full source code for simulations, examples, and the 401(k) application used in the paper. Thank you for your interest and we hope you find it useful.

## Citation
If you use the code or the results of our article and/or this repo in your work, please cite the following entry:
```bibtex
@article{dance2025cocycles,
  title   = {Counterfactual Cocycles: A Framework for Robust and Coherent Counterfactual Transports},
  author  = {Dance, Hugh and Bloem-Reddy, Benjamin},
  journal = {arXiv preprint arXiv:2405.13844},
  year    = {2025},
  url     = {https://arxiv.org/abs/2405.13844}
}
```

## Installation

To get started, please follow the installation instructions below.

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

To run the examples and application, you will need ```ipykernel``` to be able to run the .ipynb notebooks
```bash
pip install notebook ipykernel
```
(Optional) Register this env as a selectable Jupyter kernel
```bash
python -m ipykernel install --user --name=cocycles --display-name "cocycles"
```
## Repository Structure
Below we show the basic structure of the repo. The source code is in the ```causal_cocycle``` folder. Code to run the ```simulations```, ```examples``` and ```applications``` is in those folders respectively. 
```bash
Cocycles/
├── simulations/                    # .py runners for simulation experiments (and *_hpc.py for clusters)
│   ├── linear_model/               # code for Experiment 8.1
│   ├── OT/                         # code for Experiment 8.2
│   └── Csuite/                     # code for Experiment 8.3
├── examples/                       # notebooks and code for figures and examples
│   ├── scm_example/                # code for Example 1
│   ├── ot_example/                 # code for Example 2
│   └── illustrative examples/      # code for illustrative figures
├── applications/                   # code for 401k example
├── causal_cocycle/                 # core library (models, training, kernels)
├── environment.yml
├── requirements.txt
├── setup.py
└── README.md
```

## Reproducing Experiments in the Paper

### Examples
Code for Examples 1 and 2 in the paper can be found in the ```examples``` folder in Jupyter notebooks. To run these, you will need to first open a ```jupyter notebook```.

- To replicate the causal normalizing flows on the binary noise SCM in Example 1, run ```gaussian_flow_binary_example.ipynb``` and ```laplace_flow_binary_example.ipynb``` in the ```scm_example``` subfolder
- To replicate the causal normalizing flows on the mixed tailed noise SCM in Example 1, run ```gaussian_flow_mixed_example.ipynb``` and ```laplace_flow_mixed_example.ipynb``` in the ```scm_example``` subfolder
- To replicate the Gaussian Optimal transport Example 2 and Figure 2, run ```OT_inconsistency.ipynb``` in the ```ot_example``` subfolder
- To replicate counterfactual cocycles on the binary noise SCM and mixed tailed noise SCM in Example 1, run ```cocycles_binary_example.ipynb``` and ```cocycles_mixedtails_example.ipynb``` respectively in the ```scm_example``` subfolder.

### Simulations

**Experiment 8.1 (Noise Ablation in Linear Model)**

For the simulations on the linear model with cross-validation over flow architectures (i.e., Table 3), run the following files:
```bash
python simulations/linear_model/run_simlin_cocycles.py # for cocycles (CMMD-V/CMMD-U)
python simulations/linear_model/run_simlin_bgm.py # for bijective causal models with different base distributions
```
For the simulations on the linear model with the fxed linear architecture (i.e., Figure 10), run the following file:
```bash
python simulations/linear_model/run_simlin_linearfixed.py # Run cocycles, maximum-likelihood BGMs and URR BGMs with fixed linear architecture
```

** Experiment 8.2 (Confounding and Path-Consistency Ablation)**

For the simulations on the confounded chain DAG, run the following files:
```bash
python simulations/OT/run_simot_chain_cocycles.py # for cocycles (MAF flow)
python simulations/OT/run_simot_chain_ot.py # for OT and sequential OT
```

For the simulations on the non-additive triangle DAG, run the following files:
```bash
python simulations/OT/run_simot_cocycles.py # for cocycles (MAF flow)
python simulations/OT/run_simot_ot.py # for OT and sequential OT
```

** Experiment 8.3 (SCM Benchmarks)**

For the simulations on SCM benchmarks, run the following files:
```bash
python simulations/Csuite/run_simcsuite_cocycles.py # for cocycles
python simulations/Csuite/run_simcsuite_carefl.py # for CAREFL
python simulations/Csuite/run_simcsuite_causalnf.py # for CAUSALNF
python simulations/Csuite/run_simcsuite_bgm.py # for BGM
```


### 401(k) application
To replicate the results of the 401(k) application open a ```jupyter notebook``` and run ```e401k-Cocycles-NF.ipynb``` in the ```applications``` folder.

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact:

Hugh W. Dance,
PhD Researcher, Machine Learning,
Gatsby Computational Neuroscience Unit, UCL
uctphwd@ucl.ac.uk
