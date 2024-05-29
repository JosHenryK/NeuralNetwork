import random
import json

with open("settings.json") as settings_typehead:
    settings_dict = json.load(settings_typehead)

def generate_random_array(num_of_items):
    temp_array = []
    for x in range(0 , num_of_items):
        temp_array.append(random.randint(-10 , 10))
    return temp_array

def generate_weight_file_struct():
    initial_hidden_layer_array = []
    for x in range(0 , settings_dict["number of neurons per hidden layer"]):
        initial_hidden_layer_array.append(generate_random_array(settings_dict["number of input neurons"]))

    hidden_layer_array = []
    for x in range(0 , settings_dict["number of neurons per hidden layer"]):
        hidden_layer_array.append(generate_random_array(settings_dict["number of neurons per hidden layer"]))

    output_layer_array = []
    for x in range(0 , settings_dict["number of output neurons"]):
        output_layer_array.append(generate_random_array(settings_dict["number of neurons per hidden layer"]))

    weight_file_struct = []
    weight_file_struct.append(initial_hidden_layer_array)
    for x in range(0 , (settings_dict["number of hidden layers"] - 1)):
        weight_file_struct.append(hidden_layer_array)
    weight_file_struct.append(output_layer_array)

    return weight_file_struct

def generate_bias_file_struct():
    hidden_layer_array = generate_random_array(settings_dict["number of neurons per hidden layer"])
    output_layer_array = generate_random_array(settings_dict["number of output neurons"])

    bias_file_struct = []
    for x in range(0 , settings_dict["number of hidden layers"]):
        bias_file_struct.append(hidden_layer_array)
    bias_file_struct.append(output_layer_array)

    return bias_file_struct

def calc_activation_function(activation):
    e = 2.718282
    return (1 / (1 + (1 / pow(e , activation))))

def calc_inverse_activation_function(activation):
    e = 2.718282
    return (activation * (1 + (1 / pow(e , activation))))

def calc_layer_activation(incoming_activation , weights , biases):
    running_activation_array = []
    for x in range(0 , len(biases)):
        running_activation = 0
        for y in range(0 , len(incoming_activation)):
            running_activation += (incoming_activation[y] * weights[x][y])
        running_activation_array.append(calc_activation_function(running_activation + biases[x]))
    return running_activation_array

def calc_network_activation(input_array , weights , biases):
    network_activation = []
    running_activation = []
    running_activation = calc_layer_activation(input_array , weights[0] , biases[0])
    network_activation.append(running_activation)
    for x in range(1 , (settings_dict["number of hidden layers"] + 1)):
        running_activation = calc_layer_activation(running_activation , weights[x] , biases[x])
        network_activation.append(running_activation)
    return network_activation

def calc_uncompressed_network_activation(input_array , weights , biases):
    network_activation = calc_network_activation(input_array , weights , biases)
    uncompressed_network_activation = calc_network_activation(input_array , weights , biases) #will get overwritten, just to initialize array dimenion structure
    for x in range(0 , len(network_activation)):
        for y in range(0 , len(network_activation[x])):
            uncompressed_network_activation[x][y] = calc_inverse_activation_function(network_activation[x][y])
    return uncompressed_network_activation

def calc_output_activation(input_array , weights , biases):
    network_activation = calc_network_activation(input_array , weights , biases)
    return network_activation[settings_dict["number of hidden layers"]]

def calc_network_error(certainty_value_array , correct_identification):
    running_error = 0
    certainty_name_array = settings_dict["output neuron names"]
    for x in range(0 , settings_dict["number of output neurons"]):
        if (certainty_name_array[x] == correct_identification):
            running_error += pow((certainty_value_array[x] - 1) , 2)
        else:
            running_error += pow((certainty_value_array[x] - 0) , 2)
    return running_error

def calc_derivative_activation_function(activation):
    e = 2.718282
    return (pow(e , -activation) / pow((1 + pow(e , -activation)) , 2))

def train_network(training_folder_name , weights , biases):
    input_array = [0.1 , 0.2] #derived from training file
    uncompressed_network_activation_array = calc_uncompressed_network_activation(input_array , weights , biases)
    network_activation = calc_network_activation(input_array , weights , biases)

    weight_gradient_array = generate_weight_file_struct() #will get overwritten, just to initialize array dimenion structure
    bias_gradient_array = generate_bias_file_struct() #will get overwritten, just to initialize array dimenion structure
    for x in range(settings_dict["number of hidden layers"] , -1 , -1): #iterate layer
        for y in range(0 , len(biases[x])): #iterate neuron
            for z in range(0 , len(weights[x][y])): #iterate weight
                partial_cost_over_partial_weight = network_activation[x - 1][y] * calc_derivative_activation_function(uncompressed_network_activation_array[x][y]) * 2 * (network_activation[x][y] - 1)
                weight_gradient_array[x][y][z] = partial_cost_over_partial_weight
            partial_cost_over_partial_bias = calc_derivative_activation_function(uncompressed_network_activation_array[x][y]) * 2 * (network_activation[x][y] - 1)
            bias_gradient_array[x][y] = partial_cost_over_partial_bias

    with open(settings_dict["data file name"] , "r") as data_file_typehead:
        data_file_dict = json.load(data_file_typehead)
    for x in range(0 , len(biases)):
        for y in range(0 , len(biases[x])):
            data_file_dict["biases"][x][y] = biases[x][y] + bias_gradient_array[x][y]
            for z in range(0 , len(weights[x][y])):
                data_file_dict["weights"][x][y][z] = weights[x][y][z] + weight_gradient_array[x][y][z]
    with open(settings_dict["data file name"] , "w") as data_file_typehead:
        data_file_typehead.write(json.dumps(data_file_dict))