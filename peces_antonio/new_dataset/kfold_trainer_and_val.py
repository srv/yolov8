from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
from natsort import natsorted
import random
import time
from datetime import datetime, timedelta
import json
import pandas as pd
import time

if __name__ == "__main__":
    device = "0"

    imgsz = 300
    batch = 7
    epochs = 1
    patience = 0
    optimizer = 'SGD'
    lr0 = 0.001
    cfg_path = r"/home/antonio/yolov8/peces_antonio/configs/best_da_modified.yaml"

    path_to_project = fr"/home/antonio/yolov8/peces_antonio/new_dataset/validation_test"
    if not os.path.exists(path_to_project): 
        os.makedirs(path_to_project)

    path_to_dataset = r"/home/antonio/yolov8/peces_antonio/new_dataset/dataset"
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
            "n": "nano",
            # "s": "small",
            # "m": "medium",
            # "l": "large",
            # "x": "extra_large"
        }
        k = 5  # num folds
        # Tidy train-val splits from k-fold
        ds_path = os.path.join(path_to_dataset, "folds")
        
        # create temp train and val (or empty them)
        print("DS PATH: ", ds_path)
        # create the k fold iteration (here to avoid doing it every time)
        # # k fold 
        for i in range(k, k+1):
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
                project_name = os.path.join(path_to_project, model_sizes[model_size])
                    
                run_name = os.path.join(project_name, "fold_"+str(i))
                
                model = YOLO(f"yolov8{model_size}-seg.pt")
                
                train_dict = dict(
                    data=dataset_yaml,
                    epochs=epochs, 
                    patience=patience, 
                    batch=batch, 
                    imgsz=imgsz,
                    device=device, 
                    project=project_name, 
                    name=f"fold_{i}",
                    optimizer=optimizer,
                    lr0=lr0,
                    seed=seed,
                    cfg=cfg_path
                )
                
                model.train(**train_dict)
                del model
                            
                with open(os.path.join(project_name, f"fold_{i}", "train_args.json"), "w") as json_file:
                    json.dump(train_dict, json_file, indent=4)
                    
                torch.cuda.empty_cache()
                # print("Sleeping 30 secs!")
                # time.sleep(30)
                
                model = YOLO(os.path.join(project_name, f"fold_{i}", "weights", "best.pt"))
                results = model.val(
                    data=dataset_yaml, 
                    imgsz=imgsz, 
                    batch=batch, 
                    # save_json=True,
                    device=device,
                    split="val", 
                    project=os.path.join(project_name, f"fold_{i}"),
                    name="val" 
                )
                
                
                results_data = {}
                # Keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']
                for key, value in zip(results.keys, results.mean_results()):
                    results_data[key] = value

                #TODO: Store also fitness values (?)
                
                print(json.dumps(results_data, indent=4))
                
                with open(os.path.join(project_name, f"fold_{i}", "val", "validation_results.json"), "w") as json_file:
                    json.dump(results_data, json_file, indent=4)
                
                print('Toy durmiendo ._. zzZ')
                sleep_time = 0.1 #min
                print(f"Me despierto a las {(datetime.now() + timedelta(minutes=sleep_time)).strftime('%H:%M:%S')} :(")
                time.sleep(sleep_time*60)

    # Compute mean value (fold metrics)
    keys = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']

    # Results init
    mean_data = {key:0 for key in keys}

    for fold_idx in range(1, k+1):
        fold_data_path = os.path.join(project_name, f"fold_{fold_idx}", "val", "validation_results.json")
        with open(fold_data_path, 'r') as file: 
            fold_data = json.load(file)
            
        for key in keys:
            mean_data[key] += fold_data[key]
        mean_data["F1(B)"] = 2 * (mean_data['metrics/recall(B)'] * mean_data['metrics/precision(B)']) / (mean_data['metrics/recall(B)'] + mean_data['metrics/precision(B)'])
        mean_data["F1(M)"] = 2 * (mean_data['metrics/recall(M)'] * mean_data['metrics/precision(M)']) / (mean_data['metrics/recall(M)'] + mean_data['metrics/precision(M)'])


    keys.append("F1(B)")
    keys.append("F1(M)")
    mean_data = {key: mean_data[key]/k for key in keys}

    df = pd.DataFrame.from_dict([mean_data], orient="columns")
    df.to_csv(os.path.join(project_name, "results.csv"), index=False)