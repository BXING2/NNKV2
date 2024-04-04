# class for params
import os
import sys
import shutil
import random
import numpy as np

class params:

    def __init__(self):
        '''
        Attributes:
        attempt_frequency: attempt jump frequency, float value;
        boltzmann_constant: boltzmann constant, float value;
        params_inp_name: user input file name, string;
        dim_row: line number of simulation box dimensions in lammps dump file, int;
        dim_row_num: number of lines for simulation box dimensions in lammps dump file, int;
        config_row: line number of atomic information in lammps dump file, int;
        num_of_atoms: number of atoms, int;
        init_step: initial jump step, int;
        num_of_steps: number of jump steps, int;
        random_vacancy: generate a vacancy id randomly, int(boolean);
        vacancy_id: user specified vacancy id, int;
        dump_vacancy_id: dump vacancy id, int(boolean);
        num_of_cpus: number of cpus, int;
        flatten: flatten local neuron maps, int(boolean);
        cutoff: cutoff distance, float;
        voxel_size: voxel size for building up neuron map, float;
        temperature: simulation temperature, float;
        init_config_dump: initial dump file name, string;
        ml_model_weight: deep neural network weights file name, string;
        log_file: log file name, string;
        res_dir: output directory name, string;
        dims: simulation box dimensions, array;
        init_config: initial atomic configuration, array;
        config: current atomic configuration, array;
        box_lengths: simulation box length, array;
        path_vects: vectors indicating diffusion path directions, array;
        local_neuron_map_dims: shape of local neuron map, list;
        local_neuron_map_center_index: index of local neuron map center, list;
        model_weight: deep neural network weights, array;
        f: file for storing jump atom information including jump id/type/index;


        Method:
        add_param: add attributes to the params class;
        parse_inp: load user input parameters;
        parse_config: load atomic information;
        load_path_vects: generate and load path vectors;
        '''

        # predefined params
        self.attempt_frequency = 1e13
        self.boltzmann_constant = 8.617333e-5
        self.params_inp_name = sys.argv[1]
        
        # load user input params 
        key_int = {"dim_row", "dim_row_num", "config_row", "num_of_atoms", \
                   "init_step", "num_of_steps", \
                   "random_vacancy", "vacancy_id", "dump_vacancy_id", \
                   "num_of_cpus", "flatten"}

        key_float = {"cutoff", "voxel_size", "temperature"}

        key_str = {"init_config_dump", "ml_model_weight", "log_file", "res_dir"}
        
        for key, val in self.parse_inp():
            if key in key_int:
                val = int(val)
            if key in key_float:
                val = float(val)
            self.add_param(key=key, val=val)
        
        # generate a random vacancy
        if self.random_vacancy == True:
            self.vacancy_id = random.randint(1, self.num_of_atoms)

        # load configs 
        self.dims, self.init_config = self.parse_config()
        self.dims, self.init_config = self.dims.round(2), self.init_config.round(2)
        self.config = self.init_config.copy()
        self.config[self.vacancy_id-1, 1] = 0
        self.box_lengths = self.dims[:, 1] - self.dims[:, 0]
        
        # load vects of diffusion paths
        self.path_vects = self.load_path_vects()
        
        # local neuron map dims
        self.local_neuron_map_dims = [2 * np.round(self.cutoff / self.voxel_size - 0.5).astype(np.int32) + 1 for _ in range(3)]
        self.local_neuron_map_center_index = [dim // 2 for dim in self.local_neuron_map_dims]

        # load model weights
        self.model_weight = np.load(self.ml_model_weight, allow_pickle=True)
        
        # create directory for storing results
        if os.path.exists(self.res_dir):
            shutil.rmtree(self.res_dir)
        os.mkdir(self.res_dir)
        
        # open log file
        self.f = open(os.path.join(self.res_dir, self.log_file), "w")

    def add_param(self, key, val):
        setattr(self, key, val)
    
    def parse_inp(self):
        f = open(self.params_inp_name, "r")
        lines = [line.replace(" ", "").strip().split(":") for line in f.readlines() if line != "\n"]
        return lines
    
    def parse_config(self):
        dims = np.loadtxt(self.init_config_dump, delimiter=" ", skiprows=self.dim_row-1, max_rows=self.dim_row_num)
        configs = np.loadtxt(self.init_config_dump, delimiter=" ", skiprows=self.config_row-1, max_rows=self.num_of_atoms)

        return dims, configs

    def load_path_vects(self):

        path_vects = []
        for x in [1, -1]:
            for y in [1, -1]:
                for z in [1, -1]:
                    path_vects.append([x, y, z])

        return np.array(path_vects)
    
