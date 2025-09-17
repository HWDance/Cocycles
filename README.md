# Counterfactual Cocycles
Code for simulations and application in [Counterfactual Cocycles](https://arxiv.org/abs/2405.13844) (Dance and Bloem-Reddy, 2025).

# 


## Installation (from scratch)

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

## Citation
@article{dance2025counterfactualcocycles,
  title   = {Counterfactual Cocycles: A Framework for Robust and Coherent Counterfactual Transports},
  author  = {Dance, Hugh and Bloem-Reddy, Benjamin},
  journal = {Journal of Machine Learning Research},
  year    = {2025}
}

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

## Contact
For questions or feedback, please contact:

Hugh W. Dance,
PhD Researcher, Machine Learning,
Gatsby Computational Neuroscience Unit, UCL
uctphwd@ucl.ac.uk
