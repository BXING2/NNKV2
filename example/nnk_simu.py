# neural network kinetics scheme program

# import packages
import time
import random
import numpy as np

import os
import sys

import nnk.params
import nnk.neuron_map
import nnk.kmc_module
import nnk.ml_predict
import nnk.utils

def main():
    
    # initialize params class 
    nnk_params = nnk.params.params()

    # initialize neuron maps
    nnk_neuron_map = nnk.neuron_map.neuron_map(nnk_params)

    # save id to type/index map
    np.save(os.path.join(nnk_params.res_dir, "init_map.npy"), nnk_neuron_map.id_to_type_index)

    # initialize kinetics module 
    kmc_kinetics = nnk.kmc_module.kmc(nnk_params, nnk_neuron_map)

    # dump vacancy
    if nnk_params.dump_vacancy_id == True:
        nnk.utils.dump_id(nnk_params.vacancy_id, 0, nnk_params.f)
    
    # simulate diffusion   
    for step in range(nnk_params.init_step, nnk_params.num_of_steps + nnk_params.init_step):
        
        # update local neuron map
        local_id_neuron_map, local_type_neuron_map, neigh_ids = nnk_neuron_map.create_local_neuron_map()
        
        # aggragate local neuron maps 
        neuron_map_vects = nnk_neuron_map.aggregate_local_neuron_map()
        
        # neural network prediction 
        energy_barriers = nnk.ml_predict.predict(nnk_params.model_weight, neuron_map_vects)
        
        # neuron kinetics 
        kmc_inp = [neigh_ids, energy_barriers]
        jump_id, jump_time = kmc_kinetics.execute_kmc(kmc_inp) 
        
        # dump jump id/time
        nnk.utils.dump_id(jump_id, jump_time, nnk_params.f)
   
    nnk_params.f.close()

    return 0

if __name__ == "__main__":
    
    main()
