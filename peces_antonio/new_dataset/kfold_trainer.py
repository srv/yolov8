from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
from natsort import natsorted
import random
import time
from datetime import datetime, timedelta

imgsz = 640
batch = 5
epochs = 300
patience = 100

path_to_project = r"D:\yolov8\peces_antonio\new_dataset\kfold_large_640_default"
if not os.path.exists(path_to_project): 
    os.makedirs(path_to_project)

path_to_dataset = r"D:\yolov8\peces_antonio\new_dataset\dataset"
txt_path = os.path.join(path_to_project, "calls.txt")
dataset_yaml = os.path.join(path_to_dataset, "data_5_fold.yaml")


tmp_splits = ["train","valid"]
file_types = ["images","labels"]

def create_empty_temp_dirs(base_path):
    print("Base path is:", base_path)
    for tmp_split in tmp_splits:
        for file_type in file_types:
            tmp_dir = os.path.join(base_path, tmp_split,file_type)
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            else:
                shutil.rmtree(tmp_dir)
                os.makedirs(tmp_dir)


do_train = True
folds_created = True
k = 5
seed=42
check_imgs_array,check_lbls_array=[],[]
random.seed(seed)

if not folds_created:
    # PART 1: Create k-folds
    # 1.1 Create data_dict
    folders = ["train", "valid"]
    images, labels = [], []
    for folder in folders:
        images.extend(glob.glob(os.path.join(path_to_dataset, folder, "images","**")))
        labels.extend(glob.glob(os.path.join(path_to_dataset, folder, "labels","**")))
        if len(images) != len(labels):
            print("WARNING!!!: THE NUMBER OF IMAGES AND LABELS DOES NOT MATCH")
        print("num images : ", len(images))

    images = natsorted(images)
    labels = natsorted(labels)
    train_val_data = dict(zip(images, labels))

    # 1.2 Create folds
    folds_path = os.path.join(path_to_dataset, "folds")
    for i in range(1, k+1):
        os.makedirs(os.path.join(folds_path, str(i), "images"), exist_ok=True)
        os.makedirs(os.path.join(folds_path, str(i), "labels"), exist_ok=True)

    # 1.3 Copy images and labels into folds
    init_idx = 0
    
    random.shuffle(images)
    final_idx = int(len(images) / k) + 1  # the last elem (b) is not included in [a:b] function
    print("\n num of train images is: ", len(images), "\n")
    print("idx is: ", final_idx, "\n")

    for i in range(1, k+1):
        print("Copying images to the ", i, "th fold")
        print(" init idx is: ", init_idx, "\n", "final idx is: ", final_idx, "\n")

        for sample_idx, sample in enumerate(images[init_idx:final_idx]):
            shutil.copyfile(sample, os.path.join(folds_path, str(i), "images", os.path.split(sample)[-1]))
            shutil.copyfile(train_val_data[sample], os.path.join(folds_path, str(i), "labels", os.path.split(train_val_data[sample])[-1]))
            check_imgs_array.append(sample)
            check_lbls_array.append(train_val_data[sample],)

        init_idx = final_idx
        final_idx = final_idx + int(len(images) / k) + 1
        if i == k - 1:
            final_idx = len(images)

if len(set(list(check_imgs_array))) != len(check_imgs_array) or len(set(list(check_lbls_array))) != len(check_lbls_array):
    print("WARNING: SOMETHING HAS BEEN COPIED MORE THAN ONE TIMEEEE!!!!!!!!!!!!!!")

# FOLDS CREATED

if do_train:
    model_sizes = {
        # "n": "nano",
        # "s": "small",
        # "m": "medium",
        "l": "large",
        # "x": "extra_large"
    }
    lrs=[0]
    k = 5  # num folds
    for lr in lrs:
        # Tidy train-val splits from k-fold
        ds_path = os.path.join(path_to_dataset, "folds")
        
        # create temp train and val (or empty them)
        print("DS PATH: ", ds_path)
        # create the k fold iteration (here to avoid doing it every time)
        # # k fold 
        for i in range(1, k+1):
            create_empty_temp_dirs(ds_path)
            for f in range(1, k+1):
                if f == i:
                    print("copying val files to: ", os.path.join(ds_path, "valid"))
                    for img, lbl in zip(glob.glob(os.path.join(ds_path, str(f), "images","*")), glob.glob(os.path.join(ds_path, str(f), "labels","*"))):
                        shutil.copyfile(img, os.path.join(ds_path, "valid","images", os.path.split(img)[-1]))
                        shutil.copyfile(lbl, os.path.join(ds_path, "valid","labels", os.path.split(lbl)[-1]))
                else:
                    print("copying train files to ", os.path.join(ds_path, "train"))
                    for img, lbl in zip(glob.glob(os.path.join(ds_path, str(f), "images","*")), glob.glob(os.path.join(ds_path, str(f), "labels","*"))):
                        shutil.copyfile(img, os.path.join(ds_path, "train","images", os.path.split(img)[-1]))
                        shutil.copyfile(lbl, os.path.join(ds_path, "train","labels", os.path.split(lbl)[-1]))

            for model_size in model_sizes.keys():
                project_name = os.path.join(path_to_project, model_sizes[model_size], "lr_"+str(lr))
                    
                run_name = os.path.join(project_name, "fold_"+str(i))
                
                instruction = f"python ../train_yolov8.py  \
                    --model_size {model_size} --dataset {dataset_yaml} \
                        --epochs {epochs} --batch {batch} --patience {patience} --yolo_proj {project_name} --yolo_name fold_{i} \
                            --seed {seed} --imgsz {imgsz}" 
                
                with open(txt_path, 'a+') as f:
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
