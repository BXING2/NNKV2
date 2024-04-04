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

Commands:  python nnk_simu.py user_inp

The program will generate the output files in the folder called res_data. The python script postprocess.py is used for extracting useful information from simulation outputs such as reconstructing and dumping atomic configurations. 

# Performance
![image](https://github.com/BXING2/NNKV2/assets/126745914/abe8f236-0a09-4f46-ba43-1e9858f65226)

The figure illustrates the computational time for performing NNK simulations with 100,000 atomic jumps in models with various size. The x axis indicates the number of unit cells along each dimension. When the number unit cells are 10, 20, 40, 80, 160 along each dimension, the model contains 2,000, 16,000, 128,000, 1,024,000 and 8,192,000 atoms, respectively. The initialization is only performed once even if the time increases with increasing number of atoms. Notably, the iteration time is almost independent of model size as indicated by the constant computational iteration time in different models. For long time simulations, the iteration stage acts as the bottleneck of the simulation. Based on the performance, the NNK simulation can generalize well to large scale simulations.

The simulation is performed on the Macbookm Pro (version 12.4) with a single cpu core. The average computational time for one atomic jump is around 1.6-1.7 ms.

# Example
https://github.com/BXING2/NNKV2/assets/126745914/d3cf7150-b460-481b-8a89-047a96efde1e

