import os
import yaml
from itertools import product

# # Default Hyperparameters ------------------------------------------------------------------------------------------------------
# lr0: 0.01  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
# hsv_h: 0.015  # (float) image HSV-Hue augmentation (fraction)
# hsv_s: 0.7  # (float) image HSV-Saturation augmentation (fraction)
# hsv_v: 0.4  # (float) image HSV-Value augmentation (fraction)


def generate_yaml(path: str, name: str, data: dict): 
    if not os.path.exists(path): 
        f'Path do not exist: {path}'
        os.makedirs(path)
        f'Path created.'

    with open(os.path.join(path, f'{name}.yaml'), 'w') as yaml_file:
        yaml.dump(data, yaml_file)
    print('YAML generated.')

def combination_generator(hyps_space: dict):
    combinations = list(product(*hyps_space.values()))
    while combinations != []:
        combination = combinations.pop()
        combination_dict = {
            'lr0': combination[0],
            'hsv_h': combination[1], 
            'hsv_s': combination[2], 
            'hsv_v': combination[3], 
        }
        yield combination_dict


# Creating a GRID
# hyps = {
#     'lr0': [0.001],
#     'hsv_h': [0.01, 0.015, 0.02], 
#     'hsv_s': [0.6, 0.7, 0.8], 
#     'hsv_v': [0.35, 0.4, 0.45]
# }

