import os
from ultralytics import YOLO
from generate_hyp import combination_generator, generate_yaml

project_name = "hyps_study"
project_path = ""
data_path = ""
model_size = "x"
batch = 8
seed=42
epochs = 500
patience = 0
lr = 0.01
optimizer = "SGD"

hyps = {
        'hsv_h': [0.01, 0.015, 0.02], 
        'hsv_s': [0.6, 0.7, 0.8], 
        'hsv_v': [0.35, 0.4, 0.45]
    }
combinations = combination_generator(hyps)


for combination in combinations: 
    generate_yaml(project_path, 'temp_cfg', combination)
    cfg_path = os.path.join(project_path, "temp_cfg.yaml")
    
    yolo_name = ""
    for key, value in combination.items():
        yolo_name += f'_{key}_{value}'

    instruction = f"python ../clearml_log_yolov8.py --project_name {project_name} --task_name {yolo_name} \
                        --model_size {model_size} --dataset {data_path} \
                            --epochs {epochs} --batch {batch} --patience {patience} --yolo_proj {project_path} --yolo_name {yolo_name} \
                                --seed {seed} --lr {lr} --optimizer {optimizer} --config {cfg_path}" 