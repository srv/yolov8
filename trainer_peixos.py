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

path_to_project = "/mnt/c/Users/haddo/yolov8/peixos/"
path_to_dataset_base = "/mnt/c/Users/haddo/yolov8/datasets/Instance_con_SAM_"
tmp_suffixes = ["train/images/", "train/labels/", "valid/images", "valid/labels"]


def create_empty_temp_dirs(base_path):
    print("Base path is:", base_path)
    for tmp_suffix in tmp_suffixes:
        tmp_dir = os.path.join(base_path, tmp_suffix)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        else:
            shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)

# CREATE FOLDS
# ds_versions = [5, 11, 16]

ds_versions = [16]
do_train = True
folds_created = True
k = 5
seed=42

for v in ds_versions:
    path_to_dataset = path_to_dataset_base + str(v) + "/"
    if not os.path.exists(path_to_dataset):
        os.mkdir(path_to_dataset)

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
        folds_path = os.path.join(path_to_dataset, "folds/")
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
                final_idx = len(images) - 1

# FOLDS CREATED

if do_train:
    # 2 Instructions
    train_instruction = "yolo segment train cfg={} data={} model={} epochs=200 imgsz=640 seed={}  lr0={} project={} name={}"
    val_instruction = "yolo segment val data={} model={}  project={} name={} split=val"
    # test_instruction = "yolo segment val data={} model={} project={} name={} split=test"

    lrs = [0.03, 0.01, 0.0033, 0.00011, 0.00037]
    # model_sizes = {"yolov8m-seg.pt": "medium", "yolov8l-seg.pt": "large"}
    model_sizes = {"yolov8n-seg.pt": "nano"}
    configs=["/mnt/c/Users/haddo/yolov8/ultralytics/yolo/cfg/da.yaml"]
    # configs=["/mnt/c/Users/haddo/yolov8/ultralytics/yolo/cfg/da.yaml","/mnt/c/Users/haddo/yolov8/ultralytics/yolo/cfg/no_da.yaml"]

    batch = 12
    k = 5  # num folds
    for ds_v in ds_versions:
        # Tidy train-val splits from k-fold
        ds_path = path_to_dataset_base + str(ds_v) + "/folds/"
        
        # create temp train and val (or empty them)
        print("DS PATH: ", ds_path)
        create_empty_temp_dirs(ds_path)
        # create the k fold iteration (here to avoid doing it every time)
        dataset_yaml = path_to_dataset_base + str(ds_v) + "/fish_" + str(ds_v) + ".yaml"
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
                    for config in configs: #DA no DA
                        da=config.split("/")[-1].split(".")[0]
                        
                        run_name=os.path.join(project_name,str(ds_v)+"_species","lr_"+str(lr),da,"fold_"+str(i)+"_seed_"+str(seed))
        
                        task = Task.init(project_name='PEIXOS_16_nano', task_name=run_name)
                        
                        train_instruction = "yolo segment train cfg={} data={} model={} epochs=200 imgsz=640 seed={}  lr0={} project={} name={}"
                        train_instruction_formatted=train_instruction.format(config,dataset_yaml,model_size,str(seed),str(lr),project_name,run_name) 
                        
                        val_instruction_formatted =val_instruction.format(dataset_yaml,os.path.join(project_name,run_name,"weights/"+"best.pt"),project_name,run_name+"/validation") 
                        # test_instruction_formatted =test_instruction.format(dataset_yaml,os.path.join(project_name,run_name,"weights/"+"best.pt"),project_name,run_name+"/test") 
                        
                        with open('/mnt/c/Users/haddo/yolov8/calls_nano_16.txt', 'a+') as f:
                            f.write(train_instruction_formatted)
                            f.write("\n")
                            f.write(val_instruction_formatted)
                            f.write("\n")
                            # f.write(test_instruction_formatted)
                            f.write("------------------------------------------------------------- \n")
                            f.write("\n")

                        print(train_instruction_formatted)
                        # Use the formatted instructions
                        os.system(train_instruction_formatted)
                        os.system(val_instruction_formatted)
                        # os.system(test_instruction_formatted)
                        task.close()
                        time.sleep(300)

