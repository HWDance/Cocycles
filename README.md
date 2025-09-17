# Counterfactual Cocycles
This is the repository to reproduce the results of the article [Counterfactual Cocycles](https://arxiv.org/abs/2405.13844) (Dance and Bloem-Reddy, 2025). It contains all source code used in our experiments. We appreciate your interest in our work and hope you find it valuable.

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
## Reproducing Experiments 

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

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact:

Hugh W. Dance,
PhD Researcher, Machine Learning,
Gatsby Computational Neuroscience Unit, UCL
uctphwd@ucl.ac.uk
