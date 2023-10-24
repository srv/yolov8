import os
import time
from datetime import datetime, timedelta

config_files = './configs/'
dataset_yaml = '/mnt/c/Users/Uib/Documents/peces/dataset/data.yaml'
epochs = 5

for model_size in ['n', 's', 'm', 'l', 'x']: 
    for config_file in os.listdir(config_files):
        
        task_name = f'test_{model_size}'

        print('Current model size: ', model_size) 
        print('Current config file: ', config_file)
        inst = f"python clearml_log_yolov8.py --project_name 'Peces' --task_name {task_name} \
            --model_size {model_size} --dataset {dataset_yaml} --config {config_files}/{config_file}\
                --optimizer 'SGD' --epochs {epochs} --batch 8 --patience 20 \
                    --yolo_proj './train_argparse_test' --yolo_name {task_name}"
        
        os.system(inst)
        
        print('Toy durmiendo ._. zzZ')
        sleep_time = 1 #min
        print(f"Me despierto a las {(datetime.now() + timedelta(minutes=sleep_time)).strftime('%H:%M:%S')} :(")
        time.sleep(sleep_time*60)


