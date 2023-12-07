from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np
# Load a model
import matplotlib.pyplot as plt
from natsort import natsorted
# data_path="/home/uib/DATA/PEIXOS/PLOME_16ESP_OD/test/images/"
data_path="/home/uib/PLOME/stereo_tests/Andratx_2023-10-11__12-57-42_5/all/"

# model_path="/home/uib/PLOME/fish_trained_models/yolov8/binary_fish/"
model_path="/home/uib/PLOME/fish_trained_models/yolov8/16_classes"

# out_path="/home/uib/PLOME/stereo_tests/test_stereo_andratx_interior/"
out_path=data_path
model_name="yolov8lr_medium_16cIS_f2.pt"
# model_name="yolov8lr_medium_BF_f2.pt.pt"

yolo_model=os.path.join(model_path,model_name)
#yolo predict model="C:\Users\Uib\yolov8\peces_antonio\nano\fold_3\weights\best.pt" source='C:\Users\Uib\yolov8\peces_antonio\seleccion_yolanda'

# Load a pretrained YOLOv8n model
model = YOLO(yolo_model)

fish_dict={ 0: 'Chromis chromis', 1: 'Coris julis', 2: 'Dentex dentex', 3: 'Diplodus annularis', 4: 'Diplodus sargus',
            5: 'Diplodus vulgaris', 6: 'Epinephelus marginatus', 7: 'Lithognathus mormyrus', 8: 'Mugilidae prob Chelon',
            9: 'Oblada melanura', 10: 'Pomatous salator', 11: 'Sciena umbra', 12: 'Seriola dumerili',
            13: 'Serranus', 14: 'Spicara maena', 15: 'Spondyliosoma cantharus'}

num_classes=len(fish_dict.items())
print("num_classes is: ",num_classes)

# Run inference on the entire directory:
# model.predict(data_path, save=True, imgsz=1024, conf=0.4, project=out_path,name="inference_yv8mbf",retina_masks=True,line_width=5,boxes=False,show_labels=True)

for img in natsorted(os.listdir(data_path)):
    if "left" in img:
        print("running inference on ", img)
        img_path=os.path.join(data_path,img)
        results=model.predict(img_path,conf=0.4,project=out_path,name="inference_yv8m16c",retina_masks=True,line_width=5)

        print("Processing image: ",img)

        if(results[0].masks is not None):

            # Convert mask to single channel image
            # black_img=np.zeros((results[0].orig_img.shape))
            masked=np.ones((768,1024))*255
            #dictionary with model_classes:

            fish_names=results[0].names
            print(fish_names)
            fish_masks=results[0].masks
            fish_boxes=results[0].boxes

            for i,mask in enumerate(fish_masks):

                fish_cls=fish_names[int(fish_boxes.cls[i])]

                mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
                # Convert single channel grayscale to 3 channel image
                mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

                # Get the size of the original image (height, width, channels)
                h2, w2, c2 = results[0].orig_img.shape

                # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
                mask = cv2.resize(mask_3channel, (w2, h2))

                # Convert BGR to HSV
                hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

                # Define range of brightness in HSV
                lower_black = np.array([0,0,0])
                upper_black = np.array([0,0,1])

                # Create a mask. Threshold the HSV image to get everything black
                mask = cv2.inRange(mask, lower_black, upper_black)

                masked=masked*mask
                # cv2.imwrite(os.path.join(out_path,"mask_"+str(i)+".jpg"), masked)

                i+=1

            # Invert the mask to get everything but black
            mask_final = masked*(-1)+255

            cv2.imwrite(os.path.join(out_path,img.split("_")[0]+"_masked.jpg"), mask_final)
        else:
            print("no masks found for IMAGE: ",img)




