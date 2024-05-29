import json
import methods

with open("settings.json") as settings_typehead:
    settings_dict = json.load(settings_typehead)

with open(settings_dict["data file name"]) as data_file_typehead:
    initialization_status = 1
    if (data_file_typehead.readline() == ""):
        response = input("data file is not initialized, would you like to inialize it randomly? (y/n)")
        if (response == "y"):
            initialization_status = 0

#initialize weight and bias file structure
if (initialization_status == 0):
    data_file_struct = {}
    data_file_struct["weights"] = methods.generate_weight_file_struct()
    data_file_struct["biases"] = methods.generate_bias_file_struct()

    with open(settings_dict["data file name"] , "a") as data_file_typehead:
        data_file_typehead.write(json.dumps(data_file_struct))

#run the network
with open(settings_dict["data file name"]) as data_file_typehead:
    data_file_dict = json.load(data_file_typehead)
    weights = data_file_dict["weights"]
    biases = data_file_dict["biases"]

#train the network
methods.train_network("n/a" , weights , biases)