# NNKV2
This repository contains the python package called neural network kinetics (NNK) for vacancy diffusion simulation. The NNK scheme can efficiently simulate vacancy diffusion via combining deep neural network and kinetic Monte Carlo. The deep neural network predicts the path-dependent energy barriers from local atomic environment encoded by on-lattice representation. The kinetic Monte Carlo samples the diffusion jump direction and timescale based on neural network predicted energy barriers. The simulation includes an initialization and iteration stage. The initialization stage includes one-time tasks such as neuron map creation, neighbor atom search, etc. The iterative stage includes repetitive tasks such as local neuron map update, energy barrier prediction, etc.

# Installation
The python package will be installed following the listed steps.

1. Download the python package.
2. cd NNKV2
3. python setup.py sdist
4. python -m pip install .

# Usage 
The example folder provides scripts for performing NNK simulations. 

Commands
python nnk_simu.py user_inp

The program will generate the output files in the folder called res_data. The python script postprocess.py is used for extracting useful information from simulation outputs such as reconstructing and dumping atomic configurations. 

# Performance

# Example
