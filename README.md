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
## Repository Structure
Below we show the basic structure of the repo. The source code is in the ```causal_cocycle``` folder. Code to run the ```simulations```, ```examples``` and ```applications``` is in those folders respectively. 
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

## Reproducing Experiments in the Paper

### Examples
Code for Examples 1 and 2 in the paper can be found in the ```examples``` folder in Jupyter notebooks. To run these, you will need to first open a jupyter notebook:
``` bash
python -m ipykernel install --user --name=cocycles --display-name "cocycles"
jupyter notebook
```

- To replicate the causal normalizing flows on the binary noise SCM in Example 1, run ```gaussian_flow_binary_example.ipynb``` and ```laplace_flow_binary_example.ipynb``` in the ```scm_example``` subfolder
- To replicate the causal normalizing flows on the mixed tailed noise SCM in Example 1, run ```gaussian_flow_mixed_example.ipynb``` and ```laplace_flow_mixed_example.ipynb``` in the ```scm_example``` subfolder
- To replicate the Gaussian Optimal transport Example 2 and Figure 2, run ```OT_inconsistency.ipynb``` in the ```ot_example``` subfolder
- To replicate counterfactual cocycles on the binary noise SCM and mixed tailed noise SCM in Example 1, run ```cocycles_binary_example.ipynb``` and ```cocycles_mixedtails_example.ipynb``` respectively in the ```scm_example``` subfolder.


## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact:

Hugh W. Dance,
PhD Researcher, Machine Learning,
Gatsby Computational Neuroscience Unit, UCL
uctphwd@ucl.ac.uk
