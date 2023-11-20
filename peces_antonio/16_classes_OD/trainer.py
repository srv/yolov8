from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
from clearml import Task
from natsort import natsorted
import random
import time
from datetime import datetime, timedelta

seed = 42
path_to_project = r"C:\Users\Uib\yolov8\peces_antonio\16_classes_OD"
if not os.path.exists(path_to_project): 
    os.makedirs(path_to_project)
clearml_project = 'Peces_16_OD'

path_to_dataset = os.path.join(path_to_project, "dataset")
txt_path = os.path.join(path_to_project, "calls.txt")
dataset_yaml = os.path.join(path_to_dataset, "data.yaml")

model_sizes = {
        "n": "nano",
        "s": "small",
        # "m": "medium",
        # "l": "large",
        # "x": "extra_large"
    }

batch = 8

for model_size in model_sizes.keys():
    project_name = os.path.join(path_to_project, model_sizes[model_size])

    instruction = f"python ../clearml_log_yolov8.py --project_name {clearml_project} --task_name {project_name} \
                    --model_size {model_size} --dataset {dataset_yaml} \
                        --epochs 300 --batch {batch} --patience 20 --yolo_proj {project_name} --yolo_name {project_name} \
                            --seed {seed} --optimizer SGD" 

    with open(txt_path, 'a+') as f:
        f.write(instruction)
        f.write("\n")
        f.write("------------------------------------------------------------- \n")
        f.write("\n")
    
    print(instruction)            
    os.system(instruction)

    print('Toy durmiendo ._. zzZ')
    sleep_time = 5 #min
    print(f"Me despierto a las {(datetime.now() + timedelta(minutes=sleep_time)).strftime('%H:%M:%S')} :(")
    time.sleep(sleep_time*60)