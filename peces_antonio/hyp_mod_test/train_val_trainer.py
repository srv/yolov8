import os
from ultralytics import YOLO
import time 
from datetime import datetime, timedelta

project_name = "hyps_study"
project_path = r"C:\Users\haddo\yolov8\peces_antonio\hyp_mod_test"
data_path = r"C:\Users\haddo\yolov8\peces_antonio\dataset\data.yaml"
cfg_path = r"C:\Users\haddo\yolov8\peces_antonio\configs\copypaste.yaml"
model_size = "n"
batch = 8
seed=42
epochs = 5
patience = 0
lr = 0.001
optimizer = "SGD"

yolo_name = "copypaste"

instruction = f"python ../clearml_log_yolov8.py --project_name {project_name} --task_name {yolo_name} \
                    --model_size {model_size} --dataset {data_path} \
                        --epochs {epochs} --batch {batch} --patience {patience} --yolo_proj {project_path} --yolo_name {yolo_name} \
                            --seed {seed} --lr {lr} --optimizer {optimizer} --config {cfg_path}" 

os.system(instruction)

print('Toy durmiendo ._. zzZ')
sleep_time = 5 #min
print(f"Me despierto a las {(datetime.now() + timedelta(minutes=sleep_time)).strftime('%H:%M:%S')} :(")
time.sleep(sleep_time*60)


cfg_path = r"C:\Users\haddo\yolov8\peces_antonio\configs\no_da.yaml"
yolo_name = "no_da"
instruction = f"python ../clearml_log_yolov8.py --project_name {project_name} --task_name {yolo_name} \
                    --model_size {model_size} --dataset {data_path} \
                        --epochs {epochs} --batch {batch} --patience {patience} --yolo_proj {project_path} --yolo_name {yolo_name} \
                            --seed {seed} --lr {lr} --optimizer {optimizer} --config {cfg_path}" 