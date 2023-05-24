from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os

# Add the following two lines to your code, to have ClearML automatically log your experiment
from clearml import Task


train_instruction="yolo segment train data={}, model={}, epochs=200, imgsz=640,seed={}, cfg={}, lr0={},project={},name={}"    
val_instruction="yolo segment val data={}, model={} cfg={}, project={},name={} split=test"    

path_to_yamls="My path"
lrs=[0.03,0.01,0.0033,0.00011,0.00037]
model_sizes={"yolov8n.pt":"nano","yolov8s.pt":"small"}

seeds=[21,42,37,9,6]
dataset_yamls=["peixos_7.yaml","peixos_12.yaml","peixos_15.yaml"]
project_name="mi_ruta"
configs=["da.yaml","no_da.yaml"]

batch=12
for model_size in model_sizes.keys():
    project_name="mi_ruta"+model_sizes[model_size]
    #Num SPECIES:
    for dataset_yaml in dataset_yamls:
        for lr in lrs:
            #DA no DA
            for config in configs:
                for item,seed in enumerate(seeds):

                    run_name=os.path.join(project_name,"_"+dataset_yaml,"_"+lr,"_"+config,"_"+item)
                    task = Task.init(project_name='PLOME_SHALLOW', task_name=run_name)
                    train_instruction.format(dataset_yamls,model_size,str(seed),os.path.join(path_to_yamls,config),str(lr),project_name,run_name) 
                    val_instruction.format(dataset_yamls,os.path.join(project_name,run_name,"weights"+"best_weight.pt"),os.path.join(path_to_yamls,config),project_name,run_name) 

