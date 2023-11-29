import os
import shutil
import yaml

v8_dataset = r'C:\Users\Uib\yolov8\peces_antonio\binary_default\dataset'
v5_dataset = r'C:\Users\Uib\yolov8\peces_antonio\binary_default\dataset_v5'

if not os.path.exists(v5_dataset): 
    os.makedirs(v5_dataset)
    os.makedirs(os.path.join(v5_dataset, 'images'))
    os.makedirs(os.path.join(v5_dataset, 'labels'))
else: 
    raise FileExistsError(f'{v5_dataset} already exists!!')

splits = ['train', 'valid', 'test']

for split in splits: 

    for file_type in ('images', 'labels'): 
        v8_path = os.path.join(v8_dataset, split, file_type)
        v5_path = os.path.join(v5_dataset, file_type, split)
        os.makedirs(v5_path)
        for file in os.listdir(v8_path): 
            shutil.copy(os.path.join(v8_path, file), os.path.join(v5_path, file))

import yaml

with open(os.path.join(v8_dataset, 'data.yaml'), 'r') as file:
    data = yaml.safe_load(file)

data['train'] = '../images/train'
data['val'] = '../images/valid'
data['test'] = '../images/test'

with open(os.path.join(v5_dataset, 'data.yaml'), 'w') as file:
    yaml.dump(data, file, default_flow_style=False)