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

path_to_project = "/mnt/c/Users/Uib/Documents/yolov8/peces_antonio/"
path_to_dataset = "/mnt/c/Users/Uib/Documents/yolov8/peces_antonio/dataset/"
tmp_suffixes = ["train/images/", "train/labels/", "valid/images", "valid/labels"]


def create_empty_temp_dirs(base_path):
    print("Base path is:", base_path)
    for tmp_suffix in tmp_suffixes:
        tmp_dir = os.path.join(base_path, tmp_suffix)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        else:
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)


ds_versions = [16]
do_train = True
folds_created = False
k = 5
seed=42
random.seed(seed)

for v in ds_versions:

    if not folds_created:
        # PART 1: Create k-folds
        # 1.1 Create data_dict
        folders = ["train", "valid"]
        images, labels = [], []
        for folder in folders:
            images.extend(glob.glob(path_to_dataset + "/" + folder + "/images/**"))
            labels.extend(glob.glob(path_to_dataset + "/" + folder + "/labels/**"))
            if len(images) != len(labels):
                print("WARNING!!!: THE NUMBER OF IMAGES AND LABELS DOES NOT MATCH")

        images = natsorted(images)
        labels = natsorted(labels)
        train_val_data = dict(zip(images, labels))

        # 1.2 Create folds
        folds_path = os.path.join(path_to_dataset, "folds")
        for i in range(1, k+1):
            os.makedirs(folds_path + "/" + str(i) + "/images", exist_ok=True)
            os.makedirs(folds_path + "/" + str(i) + "/labels", exist_ok=True)

        # 1.3 Copy images and labels into folds
        init_idx = 0
        
        random.shuffle(images)
        final_idx = int(len(images) / k) + 1  # the last elem (b) is not included in [a:b] function
        print("\n num of train images is: ", len(images), "\n")
        print("idx is: ", final_idx, "\n")

        for i in range(1, k+1):
            print("Copying images to the ", i, "th fold")
            print(" init idx is: ", init_idx, "\n", "final idx is: ", final_idx, "\n")

            for sample in images[init_idx:final_idx]:
                shutil.copyfile(sample, folds_path + "/" + str(i) + "/images/" + sample.split("/")[-1])
                shutil.copyfile(train_val_data[sample], folds_path + "/" + str(i) + "/labels/" + train_val_data[sample].split("/")[-1])

            init_idx = final_idx
            final_idx = final_idx + int(len(images) / k) + 1
            if i == k - 1:
                final_idx = len(images)

# FOLDS CREATED

if do_train:

    # lrs = [0.03, 0.01, 0.0033, 0.00011, 0.00037]
    lrs = [0]

    model_sizes = {
        "n": "nano",
        "s": "small",
        "m": "medium",
        "l": "large",
        "x": "extra_large"
    }

    # configs=["/mnt/c/Users/haddo/yolov8/ultralytics/yolo/cfg/da.yaml"]
    # configs=["/mnt/c/Users/haddo/yolov8/ultralytics/yolo/cfg/da.yaml","/mnt/c/Users/haddo/yolov8/ultralytics/yolo/cfg/no_da.yaml"]
    batch_sizes=[8]

    
    k = 5  # num folds
    for batch in batch_sizes:
        # Tidy train-val splits from k-fold
        ds_path = path_to_dataset  + "/folds/"
        
        # create temp train and val (or empty them)
        print("DS PATH: ", ds_path)
        create_empty_temp_dirs(ds_path)
        # create the k fold iteration (here to avoid doing it every time)
        dataset_yaml = path_to_dataset +  "/data.yaml"
        # # k fold 
        for i in range(1, k+1):
            for f in range(1, k+1):
                if f == i:
                    print("copying val files to: ", ds_path + "/valid/")
                    for img, lbl in zip(glob.glob(ds_path + "/" + str(f) + "/images/*"), glob.glob(ds_path + "/" + str(f) + "/labels/*")):
                        shutil.copyfile(img, ds_path + "/valid/images/" + img.split("/")[-1])
                        shutil.copyfile(lbl, ds_path + "/valid/labels/" + lbl.split("/")[-1])
                else:
                    print("copying train files to ", ds_path + "/train/")
                    for img, lbl in zip(glob.glob(ds_path + "/" + str(f) + "/images/*"), glob.glob(ds_path + "/" + str(f) + "/labels/*")):
                        shutil.copyfile(img, ds_path + "/train/images/" + img.split("/")[-1])
                        shutil.copyfile(lbl, ds_path + "/train/labels/" + lbl.split("/")[-1])

            
            # seed=random.randint(0,100)
            for model_size in model_sizes.keys():
                project_name=path_to_project+"/"+model_sizes[model_size] +"/"  
                for lr in lrs:
                                        
                    run_name=os.path.join(project_name,"fold_"+str(i))
                    # run_name=os.path.join(project_name,"lr_"+str(lr))
                    
                    instruction = f"python ./peces_antonio/clearml_log_yolov8.py --project_name 'Pecesv8' --task_name {run_name} \
                        --model_size {model_size} --dataset {dataset_yaml} \
                            --epochs 300 --batch {batch} --patience 20 --yolo_proj {project_name} --yolo_name fold_{i} \
                                --seed {seed} --optimizer 'SGD'"
                    
                        # Also available to add --config, --lr, --optimizer

                    # task = Task.init(project_name='Peces', task_name=run_name)
                    # task.set_parameter('model_variant', model_sizes[model_size])

                    with open('/mnt/c/Users/Uib/Documents/yolov8/peces_antonio/calls_peces_antonio.txt', 'a+') as f:
                        f.write(instruction)
                        f.write("\n")
                        f.write("------------------------------------------------------------- \n")
                        f.write("\n")

                    print(instruction)
                    # Use the formatted instructions
                    os.system(instruction)
                    
                    print('Toy durmiendo ._. zzZ')
                    sleep_time = 5 #min
                    print(f"Me despierto a las {(datetime.now() + timedelta(minutes=sleep_time)).strftime('%H:%M:%S')} :(")
                    time.sleep(sleep_time*60)
