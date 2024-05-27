import json
import random

def generate_random_array(num_of_items):
    temp_array = []
    for x in range(0 , num_of_items):
        temp_array.append(random.randint(-100 , 100))
    return temp_array

def generate_weight_file_struct():
    initial_hidden_layer_array = []
    for x in range(0 , settings_dict["num of neurons per hidden layer"]):
        initial_hidden_layer_array.append(generate_random_array(settings_dict["num of input neurons"]))

    hidden_layer_array = []
    for x in range(0 , settings_dict["num of neurons per hidden layer"]):
        hidden_layer_array.append(generate_random_array(settings_dict["num of neurons per hidden layer"]))

    output_layer_array = []
    for x in range(0 , settings_dict["num of output neurons"]):
        output_layer_array.append(generate_random_array(settings_dict["num of neurons per hidden layer"]))

    weight_file_struct = []
    weight_file_struct.append(initial_hidden_layer_array)
    for x in range(0 , (settings_dict["num of hidden layers"] - 1)):
        weight_file_struct.append(hidden_layer_array)
    weight_file_struct.append(output_layer_array)

    return weight_file_struct

def generate_bias_file_struct():
    hidden_layer_array = generate_random_array(settings_dict["num of neurons per hidden layer"])
    output_layer_array = generate_random_array(settings_dict["num of output neurons"])

    bias_file_struct = []
    for x in range(0 , settings_dict["num of hidden layers"]):
        bias_file_struct.append(hidden_layer_array)
    bias_file_struct.append(output_layer_array)

    return bias_file_struct

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



# #activation function used to compress raw activation value
# def activation_function(activation):
#     e = 2.718282
#     return(1 / (1 + (1 / pow(e , activation))))

# #calculates raw activation and compresses it between 0 and 1 using activation_function()
# def calc_activiation(incoming_activation , weight, bias):
#     return(activation_function((incoming_activation * weight) + bias))