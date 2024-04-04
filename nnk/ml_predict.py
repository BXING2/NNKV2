# deep neural network predicting energy barriers

import numpy as np

def linear_combination(data, parameters):

    weights, biases = parameters
    data = np.dot(data, weights) + biases
    
    return data

def batchnormalization(data, parameters):

    gamma, beta, mean, variance = parameters
    data = (data - mean)  / (variance + 0.001) ** 0.5 
    data = gamma * data + beta  

    return data

def relu(x):
    
    x[np.where(x < 0)] = 0
 
    return x

def predict(model_weights, input_data):
        
    # neural network layer parameters including weights, bias, gamma, beta, mean, variance
    group_size = 6
    num_of_layers = len(model_weights) // group_size
    
    # hidden layers 
    for index in range(num_of_layers):
        linear_combination_output = linear_combination(input_data, model_weights[index*group_size+0: index*group_size+2])
        batchnormalization_output = batchnormalization(linear_combination_output, model_weights[index*group_size+2: (index+1)*group_size]) 
        input_data = relu(batchnormalization_output)
    
    # output layer        
    linear_combination_output = linear_combination(input_data, model_weights[num_of_layers*group_size+0: num_of_layers*group_size+2])
    
    return np.squeeze(linear_combination_output)[()]
