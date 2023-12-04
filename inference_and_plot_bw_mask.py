
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np
# Load a model
import matplotlib.pyplot as plt
from natsort import natsorted

# data_path="/home/uib/DATA/PEIXOS/PLOME_16ESP_OD/test/images/"
data_path="/home/uib/PLOME/stereo_tests/Andratx_2023-10-11__12-57-42_5/left/"

# model_path="/home/uib/PLOME/fish_trained_models/yolov8/binary_fish/"
model_path="/home/uib/PLOME/fish_trained_models/yolov8/16_classes"

# out_path="/home/uib/PLOME/stereo_tests/test_stereo_andratx_interior/"
out_path=data_path
model_name="yolov8lr_medium_16cIS_f2.pt"
# model_name="yolov8lr_medium_BF_f2.pt.pt"

# Load a pretrained YOLOv8n model
yolo_model=os.path.join(model_path,model_name)
model = YOLO(yolo_model)

#yolo predict model="C:\Users\Uib\yolov8\peces_antonio\nano\fold_3\weights\best.pt" source='C:\Users\Uib\yolov8\peces_antonio\seleccion_yolanda'
# Run inference on the entire directory:
# model.predict(data_path, save=True, imgsz=1024, conf=0.4, project=out_path,name="inference_yv8mbf",retina_masks=True,line_width=5,boxes=False,show_labels=True)


fish_dict={ 0: 'Chromis chromis', 1: 'Coris julis', 2: 'Dentex dentex', 3: 'Diplodus annularis', 4: 'Diplodus sargus',
            5: 'Diplodus vulgaris', 6: 'Epinephelus marginatus', 7: 'Lithognathus mormyrus', 8: 'Mugilidae prob Chelon',
            9: 'Oblada melanura', 10: 'Pomatous salator', 11: 'Sciena umbra', 12: 'Seriola dumerili',
            13: 'Serranus', 14: 'Spicara maena', 15: 'Spondyliosoma cantharus'}

num_classes=len(fish_dict.items())
print("num_classes is: ",num_classes)
# Define range of brightness in HSV
lower_black = np.array([0,0,0])
upper_black = np.array([0,0,1])


def generate_IS_bwmask(mask,h,w):
    mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
    # Convert single channel grayscale to 3 channel image
    mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
    # Get the size of the original image (height, width, channels)

    # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
    mask_bw = cv2.resize(mask_3channel, (w, h))
    mask_bw = cv2.inRange(mask_bw, lower_black, upper_black)

    return mask_bw

#initialize the counter of species detected in an image:
num_maks=dict(zip(fish_dict.keys(),np.zeros(len(fish_dict.keys()),dtype=int)))
class_colors=dict(zip(fish_dict.keys(),np.linspace(0,255,len(fish_dict.keys()),dtype=int)))

print("NUM MASKS: ",num_maks)
print("MASKS COLORS: ",class_colors)

#GO THROUGH THE EXTRACTED IMGS DIR
for img in natsorted(os.listdir(data_path)):
    #just inference on left image
    if "left" in img:
        print("running inference on ", img)
        img_path=os.path.join(data_path,img)
        results=model.predict(img_path,conf=0.4,project=out_path,name="inference_yv8m16c",retina_masks=True,line_width=5)

        if(results[0].masks is not None):
            # fish_names=results[0].names

            fish_masks=results[0].masks
            fish_boxes=results[0].boxes
            h, w, c = results[0].orig_img.shape
            masked=np.ones((h,w))*255
            i=0
            for i,mask in enumerate(fish_masks):
                fish_cls=int(fish_boxes.cls[i])
                print("THIS FISH IS A: ",fish_dict[fish_cls])
                mask_bw=generate_IS_bwmask(mask,h,w)

                masked=masked*mask_bw
                # cv2.imwrite("Mask_"+str(i)+".jpg", masked)
                # Invert the mask to get everything but black
                i+=1

        mask_final = masked*(-1)+255
        save_img_path=os.path.join(out_path,img+"_mask.jpg")
        cv2.imwrite(save_img_path,mask_final)
        print("Writing mask to: ",save_img_path)
            # Show the masked part of the image
