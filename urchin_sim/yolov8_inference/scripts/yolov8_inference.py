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


model_path_od = "/home/plomenuc/yolov8/trained_models/trained_models/large/best.pt"

images_path="/home/plomenuc/plome_ws/src/stereo_plome/images/"

project_path="/home/plomenuc/plome_ws/src/stereo_plome/"
out_folder="out"
# Load model
model = YOLO(model_path_od)

shape = 1024

in_images=os.listdir(images_path)


while True:

    # if os.path.exists(project_path+out_folder):
    #     shutil.rmtree(project_path+out_folder)
    #     print("Removing folder")

    # os.mkdir(project_path+out_folder)

    for image in in_images:

        # image_np = imageio.imread(os.path.join(images_path,image))
        t0 = time.time()

        # results = model([image_np])
        results=model.predict(os.path.join(images_path,image), save=True,save_conf=True,project=project_path,name=out_folder,exist_ok=True ,conf=0.5)
        # print(results[0].boxes)
        # res_plotted = results[0].plot()
        # cv2.imwrite(out_path+image+"_infered.JPG", res_plotted)

        t1 = time.time()
        t_inf = t1-t0
        print(t_inf)
    print("good night")
    time.sleep(5)
    print("good morning")



