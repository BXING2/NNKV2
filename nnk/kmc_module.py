# kinetic monte carlo simulation

import random
import numpy as np

class kmc:

    def __init__(self, params, neuron_map):

        self.params = params
        self.neuron_map = neuron_map

    def compute_jump_rates(self, energy_barriers):

        return self.params.attempt_frequency * np.exp(- np.array(energy_barriers) / (self.params.boltzmann_constant * self.params.temperature))
        
    def compute_jump_id(self, rates, rates_sum):
        
        return np.random.choice(a=self.neuron_map.neigh_ids, size=1, p=rates / rates_sum).astype(np.int32)[0]

    def compute_jump_timescale(self, rates_sum):
        
        return - np.log(random.uniform(0, 1)) / rates_sum

    def update_neuron_map(self, jump_id):
        
        # vacancy id, type, index; jump atom id, type, index
        vacancy_type, vacancy_index = self.neuron_map.id_to_type_index[self.params.vacancy_id]
        # print(self.neuron_map.id_to_type_index[jump_id])
        jump_type, jump_index = self.neuron_map.id_to_type_index[jump_id]
        
        # update id_to_type_index_map
        self.neuron_map.id_to_type_index[self.params.vacancy_id], self.neuron_map.id_to_type_index[jump_id] = \
        (vacancy_type, jump_index), (jump_type, vacancy_index)

        # update index_to_id_type map
        self.neuron_map.index_to_id_type[vacancy_index], self.neuron_map.index_to_id_type[jump_index] = \
        (jump_id, jump_type), (self.params.vacancy_id, vacancy_type)

        # update id/type neuron map
        self.neuron_map.id_neuron_map[vacancy_index], self.neuron_map.id_neuron_map[jump_index] = jump_id, self.params.vacancy_id
        self.neuron_map.type_neuron_map[vacancy_index], self.neuron_map.type_neuron_map[jump_index] = jump_type, vacancy_type

    def execute_kmc(self, kmc_inp):
        
        # load kmc inp params 
        neigh_ids, energy_barriers = kmc_inp
        
        # compute jump rates
        rates = self.compute_jump_rates(energy_barriers)
        rates_sum = np.sum(rates)
        
        # sample jump id
        jump_id = self.compute_jump_id(rates, rates_sum)
        
        # sample jump time
        jump_time = self.compute_jump_timescale(rates_sum)

        # update neuron map
        self.update_neuron_map(jump_id)

        return jump_id, jump_time
