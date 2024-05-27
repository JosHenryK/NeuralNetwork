import json
import random

def generate_random_array(num_of_items):
    temp_array = []
    for x in range(0 , num_of_items):
        temp_array.append(random.randint(-10 , 10))
    return temp_array

def generate_weight_file_struct():
    initial_hidden_layer_array = []
    for x in range(0 , settings_dict["number of neurons per hidden layer"]):
        initial_hidden_layer_array.append(generate_random_array(settings_dict["num of input neurons"]))

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

def calc_activiation(incoming_activation , weight, bias):
    return((incoming_activation * weight) + bias)

def calc_activation_function(activation):
    e = 2.718282
    return(1 / (1 + (1 / pow(e , activation))))

def calc_layer_activation(incoming_activation , weights , biases):
    outgoing_activation = []
    for x in range(0 , len(biases)):
        running_activation = 0
        for y in range(0 , len(incoming_activation)):
            running_activation += calc_activiation(incoming_activation[y] , weights[x][y] , biases[x])
        outgoing_activation.append(calc_activation_function(running_activation))
    return outgoing_activation

def calc_network_activation(input_array , weights , biases):
    running_activation = []
    running_activation = calc_layer_activation(input_array , weights[0] , biases[0])
    for x in range(1 , (settings_dict["number of hidden layers"] + 1)):
        calc_layer_activation(running_activation , weights[x] , biases[x])
    return running_activation

def display_certainties(certainty_value_array):
    certainty_name_array = settings_dict["output neuron names"]
    for x in range(0 , settings_dict["number of output neurons"]):
        print(certainty_name_array[x] , certainty_value_array[x])

#initialize weight and bias file structure
with open("settings.json") as settings_typehead:
    settings_dict = json.load(settings_typehead)
    data_file_name = settings_dict["data file name"]

with open(data_file_name) as data_file_typehead:
    initialization_status = 1
    if (data_file_typehead.readline() == ""):
        response = input("data file is not initialized, would you like to inialize it randomly? (y/n)")
        if (response == "y"):
            initialization_status = 0

if (initialization_status == 0):
    data_file_struct = {}
    data_file_struct["weights"] = generate_weight_file_struct()
    data_file_struct["biases"] = generate_bias_file_struct()

    with open(data_file_name , "a") as data_file_typehead:
        data_file_typehead.write(json.dumps(data_file_struct))

#run the network
with open(data_file_name) as data_file_typehead:
    data_file_dict = json.load(data_file_typehead)
    weights = data_file_dict["weights"]
    biases = data_file_dict["biases"]

input_array = [0.2 , 0.5]

display_certainties(calc_network_activation(input_array , weights , biases))