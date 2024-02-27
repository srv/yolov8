from ultralytics import YOLO

import imageio
import torch
import time
import cv2
from PIL import Image
from PIL import *
import numpy as np
import os
import shutil
import csv
import pandas as pd

images_path="/home/plomenuc/plome_ws/src/stereo_plome/images/"
project_path="/home/plomenuc/yolov8/inference_time_test/"
models_path="/home/plomenuc/yolov8/default_models"
out_folder="out"


csv1_filename = "/home/plomenuc/yolov8/inference_time_test/mean_inference_times_yolov8_obj.csv"
csv2_filename = "/home/plomenuc/yolov8/inference_time_test/inference_times_yolov8_obj.csv"


# Load model
shape = 1280

in_images=os.listdir(images_path)

seg_model_sizes={"nano":"yolov8n-seg.pt","small":"yolov8s-seg.pt","medium":"yolov8m-seg.pt","large":"yolov8l-seg.pt","extra-large":"yolov8x-seg.pt"}
object_model_sizes={"nano":"yolov8n.pt","small":"yolov8s.pt","medium":"yolov8m.pt","large":"yolov8l.pt","extra-large":"yolov8x.pt"}

mean_load_times=[]
mean_inf_times=[]

std_load_times=[]
std_inf_times=[]

all_load_times=[]
all_inf_times=[]

for model_size in seg_model_sizes.keys():
    inf_times=[]
    load_times=[]

    print("#################################################################################################")
    for image in in_images:

        # image_np = imageio.imread(os.path.join(images_path,image))
        model_size_path=os.path.join(models_path,seg_model_sizes[model_size])
        print("loading model: ",model_size_path)
        t0 = time.time()
        model = YOLO(model_size_path)
        t1 = time.time()
        t_load = t1-t0
        print(t_load)
        t2 = time.time()
        results=model.predict(os.path.join(images_path,image),imgsz=shape, save=True,save_conf=True,project=project_path,name=out_folder,exist_ok=True ,conf=0.5)
        t3 = time.time()
        t_inf = t3-t2
        print(t_inf)
        print("Load and inf time: ",t_load," ",t_inf)
        inf_times.append(t_inf)
        all_inf_times.append(t_inf)
        load_times.append(t_load)
        all_load_times.append(t_load)

    mean_inf_times.append(np.mean(inf_times))
    mean_load_times.append(np.mean(load_times))

    std_inf_times.append(np.std(inf_times))
    std_load_times.append(np.std(load_times))
    print("Model size is: ",model_size,", mean load time is ",np.mean(load_times), "mean inf time is: ",np.mean(inf_times))
    # Append the data for this model size to the overall data list

    # print("good night")
    # time.sleep(2)
    # print("good morning")
    print("#################################################################################################")



df2=pd.DataFrame({"load times":all_load_times,"inf_times":all_inf_times})
df2.to_csv(csv2_filename)

print(len(seg_model_sizes.keys())," ",len(mean_load_times)," ",len(mean_inf_times))

df = pd.DataFrame({'Model size': seg_model_sizes.keys(), 'Mean load times': mean_load_times,"mean inf times":mean_inf_times,'std load times': std_load_times,"std inf times":std_inf_times})


# Write DataFrame to CSV
df.to_csv(csv1_filename)



# Write data to CSV

# with open(csv_filename, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Model Size', 'Inference Times', 'Load Times'])
#     for row in data:
#         writer.writerow(row)




