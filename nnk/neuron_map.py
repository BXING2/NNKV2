# convert atomistic model to image

import time
import numpy as np
import pandas as pd
from scipy import spatial

import nnk.utils

class neuron_map:
    
    def __init__(self, params):
        '''
        Attributes:
        params: instance of params class;
        mesh_info: mesh information summary including mesh, mesh_dims, mesh_size;
        id_neuron_map: 3d array of atom ids;
        type_neuron_map: 3d array of atom types;
        id_to_type_index: dict mapping atom id to type and index;
        type_neuron_map: dict mapping atom index to id and type;
        nonzero_mask: mask indicating the index of nonzero values in local neuron map;
        neigh_index: index of first nearest neighbors;
        rotation_maps: list of eight arrays, each array contains the atom index from corresponding operation;

        Methods:
        detect_local_neuron_map_nonzero_site: find the index of nonzero values in local neuron map;
        create_local_neuron_map: return local neuron map encoding local atomistic environment(3d array);
        aggregate_local_neuron_map: return aggregated local neuron maps corresponding to eight difusion paths; 4d array with dims 8xnxnxn
        '''
        
        # params class instance
        self.params = params

        # mesh
        self.mesh_info = nnk.utils.create_mesh(box_lengths=self.params.box_lengths, voxel_size=self.params.voxel_size)
        
        # get and store initial index 
        self.id_neuron_map, self.type_neuron_map = nnk.utils.assign_atom_to_mesh(atoms=self.params.config, mesh_args=self.mesh_info)

        # build (id, type) - index maps
        self.id_to_type_index = {}
        self.index_to_id_type = {}
        for i in range(self.id_neuron_map.shape[0]):
            for j in range(self.id_neuron_map.shape[1]):
                for k in range(self.id_neuron_map.shape[2]):
                    cur_id, cur_type = self.id_neuron_map[i, j, k], self.type_neuron_map[i, j, k]
                    if cur_id != 0:
                        self.id_to_type_index[cur_id] = [cur_type, (i, j, k)]
                        self.index_to_id_type[(i, j, k)] = [cur_id, cur_type]

        # build local neuron map mask
        self.nonzero_mask = self.detect_local_neuron_map_nonzero_site()

        # build full rotation maps for all diffusion paths 
        self.neigh_index = []
        self.rotation_maps = []
        for cur_vect in self.params.path_vects:
            self.neigh_index.append(nnk.utils.compute_neigh_index(center=self.params.local_neuron_map_center_index, vect=cur_vect))
            self.rotation_maps.append(nnk.utils.rotate_mirror_data(data=self.nonzero_mask, vect=cur_vect)+ \
            np.array(self.params.local_neuron_map_center_index))
    
    def detect_local_neuron_map_nonzero_site(self):
        
        # detect nonzero value index in local neuron map
        nonzero_mask = []
        vacancy_index = self.id_to_type_index[self.params.vacancy_id][1]
        dist = lambda i, j, k: (i**2 + j**2 + k**2) ** 0.5 * self.params.voxel_size
        
        nx, ny, nz = self.mesh_info[1]
        lx, ly, lz = self.params.local_neuron_map_center_index
        for i in range(-lx, lx+1):
            for j in range(-ly, ly+1):
                for k in range(-lz, lz+1):
                    wrap_i, wrap_j, wrap_k = (i + vacancy_index[0]) % nx, (j + vacancy_index[1]) % ny, \
                    (k + vacancy_index[2]) % nz
                    
                    if (self.type_neuron_map[wrap_i, wrap_j, wrap_k] != 0) & (dist(i, j, k) < self.params.cutoff):
                        nonzero_mask.append((i, j, k))

        return np.array(nonzero_mask)        

    def create_local_neuron_map(self):
        
        # build local id/type neuron map
        vacancy_index = self.id_to_type_index[self.params.vacancy_id][1]
        lx, ly, lz = center=self.params.local_neuron_map_center_index
        self.local_id_neuron_map = np.zeros(self.params.local_neuron_map_dims)
        self.local_type_neuron_map = np.zeros(self.params.local_neuron_map_dims)
        for i, j, k in self.nonzero_mask:
            wrap_i, wrap_j, wrap_k = (i + vacancy_index[0]) % self.mesh_info[1][0], (j + vacancy_index[1]) % self.mesh_info[1][1], \
            (k + vacancy_index[2]) % self.mesh_info[1][2]
            self.local_id_neuron_map[i+lx, j+ly, k+lz] = self.id_neuron_map[wrap_i, wrap_j, wrap_k]
            self.local_type_neuron_map[i+lx, j+ly, k+lz] = self.type_neuron_map[wrap_i, wrap_j, wrap_k]
        
        # build neigh_ids 
        self.neigh_ids = [self.local_id_neuron_map[index] for index in self.neigh_index] 

        return self.local_id_neuron_map, self.local_type_neuron_map, self.neigh_ids
    
    def aggregate_local_neuron_map(self):
    
        # aggregate full local type neuron maps
        img_vects = [self.local_type_neuron_map]
        for cur_map in self.rotation_maps[1:]:
            cur_img = np.zeros_like(self.local_type_neuron_map)
            for index in range(len(self.rotation_maps[0])):
                old_i, old_j, old_k = self.rotation_maps[0][index]
                new_i, new_j, new_k = cur_map[index]
                # cur_img[old_i, old_j, old_k] = self.local_type_neuron_map[new_i, new_j, new_k]
                cur_img[new_i, new_j, new_k] = self.local_type_neuron_map[old_i, old_j, old_k]
            
            img_vects.append(cur_img)

        if self.params.flatten == True:
            img_vects = [cur_img.flatten() for cur_img in img_vects]

        return np.array(img_vects)

