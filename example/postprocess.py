# postprocess output from nnk simulation

import nnk.process_module 

map_file_name, nnk_log_file_name = "res_data/init_map.npy", "res_data/nnk.log"

neuron_map = nnk.process_module.load_map(map_file_name)
nnk_log = nnk.process_module.load_nnk_log(nnk_log_file_name)

# simulation box dimensions
dims = [64.8 for _ in range(3)]
# dims = [<dim> for _ in range(3)] # replace <dim> with real dims


# dump interval 
interval = 1

# scale or voxel size
scale = 1.62

# output all atoms when dumping configurations
# nnk.process_module.reconstruct_full_configs(neuron_map, nnk_log, interval, dims, scale, "./res_data")

# output atoms which moved during simulation when dumping configurations
nnk.process_module.reconstruct_effective_configs(neuron_map, nnk_log, interval, dims, scale, "./res_data")
