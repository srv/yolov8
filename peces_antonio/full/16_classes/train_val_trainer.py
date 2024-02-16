import os
from ultralytics import YOLO
import time 
from datetime import datetime, timedelta

best_da_cfg = r"C:\Users\haddo\yolov8\peces_antonio\hyp_study\best_da_config.yaml"
yolo_name = "16_classes"

project_path = r"C:\Users\haddo\yolov8\peces_antonio\full"
data_path = r"C:\Users\haddo\yolov8\peces_antonio\dataset_all\data.yaml"
model_size = "x"
batch = 5
seed=42
epochs = 400
patience = 0
lr = 0.001
imgsz = 1280
optimizer = "SGD"


instruction = f"python ../../train_yolov8.py \
                    --model_size {model_size} --dataset {data_path} \
                        --epochs {epochs} --batch {batch} --patience {patience} --yolo_proj {project_path} --yolo_name {yolo_name} \
                            --seed {seed} --lr {lr} --optimizer {optimizer} --config {best_da_cfg} --imgsz {imgsz} --val {False}"
    
os.system(instruction)

print('Toy durmiendo ._. zzZ')
sleep_time = 5 #min
print(f"Me despierto a las {(datetime.now() + timedelta(minutes=sleep_time)).strftime('%H:%M:%S')} :(")
time.sleep(sleep_time*60)
