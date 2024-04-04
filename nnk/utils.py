# utils functions
import numpy as np
from scipy import spatial

'''
create_mesh: create a mesh for building up neural map;
assign_atom_to_mesh: assign each atom to the corresponding mesh point based on the distance;
rotate_mirror_data: rotate/mirror local neuron map to align the current diffusion direction with the reference direction;
compute_neigh_index: compute the index of first nearest neighbors;
dump_id: write jump atom id and diffusion time to log file;
'''

def create_mesh(box_lengths, voxel_size):

    num_of_voxels = (box_lengths / voxel_size).round(0).astype(np.int32)
    mesh_size = num_of_voxels * voxel_size
    
    x = np.arange(num_of_voxels[0]) * voxel_size
    y = np.arange(num_of_voxels[1]) * voxel_size
    z = np.arange(num_of_voxels[2]) * voxel_size
    mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z)
    mesh_x, mesh_y, mesh_z = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1), mesh_z.reshape(-1, 1)
    mesh = np.c_[mesh_x, mesh_y, mesh_z].round(2)

    return mesh, tuple(num_of_voxels), mesh_size

def assign_atom_to_mesh(atoms, mesh_args, number_of_cpus=1):
    
    # load mesh information
    mesh, mesh_dims, mesh_size = mesh_args
    
    # assign atom to mesh 
    kdtree = spatial.cKDTree(data=mesh, boxsize=mesh_size)
    distance, index = kdtree.query(atoms[:, 2: 5], k=1, workers=number_of_cpus)
    
    assert np.all(distance < 1e-3) == True
    
    # create id_matrix
    id_vect = np.zeros(len(mesh))
    id_vect[index] = atoms[:, 0] 

    # create type_matrix
    type_vect = np.zeros(len(mesh))
    type_vect[index] = atoms[:, 1]

    id_arr, type_arr = id_vect.reshape(mesh_dims).astype(np.int32), type_vect.reshape(mesh_dims).astype(np.int32)

    return id_arr, type_arr

def rotate_mirror_data(data, vect):
    
    # vector to be rotated
    x, y, z = vect

    # rotation conditions
    if x >= 0:
        if y >= 0:
            theta = 0
        else:
            theta = np.pi / 2
    else:
        if y >= 0:
            theta = - np.pi / 2
        else:
            theta = np.pi

    rotation_matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    rotation_matrix = np.array(rotation_matrix)
    
    rotated_xy = np.dot(rotation_matrix, data[:, :2].T).T
    rotated_data = np.c_[rotated_xy, data[:, 2]]
    
    # mirror conditions
    if z < 0:
        rotated_data[:, 2] = - rotated_data[:, 2]
    # print(rotated_data[:10])
    # vector connecting vacancy and neighor atom always points along [111] direction;
    return rotated_data.round(0).astype(np.int32)

def compute_neigh_index(center, vect):

    return tuple([center[i]+vect[i] for i in range(len(center))])


def dump_id(jump_id, jump_time, print_f):

    print("{:.0f} {:.2e}".format(jump_id, jump_time), file=print_f)

def build_model(lattice_type, lattice_constant, num_of_cells, elements, concs, dump_file):
    
    print_f = open(dump_file, "w")
    
    dims = [num * lattice_constant for num in num_of_cells]
    
    if lattice_type == "bcc":
        basis = [[0.0, 0.0, 0.0],
                 [0.5, 0.5, 0.5]]
        basis = np.array(basis)

    x = np.arange(num_of_cells[0])
    y = np.arange(num_of_cells[1])
    z = np.arange(num_of_cells[2])
    mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z)
    mesh_x, mesh_y, mesh_z = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1), mesh_z.reshape(-1, 1)
    mesh = np.c_[mesh_x, mesh_y, mesh_z]
    
    mesh_full = []
    for cur_basis in basis:
        mesh_full.append(mesh + cur_basis)

    coords = np.concatenate(mesh_full) * lattice_constant
    coords = coords.round(2)

    atom_id = np.arange(1, len(coords) + 1).round(0)
    atom_type = np.random.choice(a=elements, size=len(atom_id), p=concs)
    print(atom_id.shape, atom_type.shape, coords.shape)
    atoms = np.c_[atom_id, atom_type, coords]

    # print prefix info
    print("ITEM: TIMESTEP", file=print_f)
    print("{:.0f}".format(0), file=print_f)
    print("ITEM: NUMBER OF ATOMS", file=print_f)
    print("{:.0f}".format(len(atom_id)), file=print_f)
    print("ITEM: BOX BOUNDS pp pp pp", file=print_f)
    for i in range(3):
        print("0 {:.2f}".format(dims[i]), file=print_f)
    print("ITEM: ATOMS id type x y z", file=print_f)

    # print atomic info
    for i in range(len(atoms)):
        print("{:.0f} {:.0f} {:.2f} {:.2f} {:.2f}".format(*atoms[i]), file=print_f)

    print_f.close()
