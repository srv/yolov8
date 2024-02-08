
from typing import final
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np
# Load a model
import matplotlib.pyplot as plt
from natsort import natsorted
import shutil
# data_path="/home/uib/DATA/PEIXOS/PLOME_16ESP_OD/test/images/"
# data_path="/home/uib/DATA/PEIXOS/LANTY/Andratx_28_06_2023/13_55/"
data_path="/home/uib/DATA/PEIXOS/LANTY/pool_plastic_fish/"

# model_path="/home/uib/PLOME/fish_trained_models/yolov8/binary_fish/"
model_path="/home/uib/yolov8/trained_models/"

# out_path="/home/uib/PLOME/stereo_tests/test_stereo_andratx_interior/"
out_path=data_path
# model_name="yolov8lr_XL_BF_f2.pt"
model_name="yolov8lr_large_16cIS_f2.pt"
model_name="plastic_fish.pt"
masks_inf_folder="inference_yv8spf"
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

fish_dict={1:"fish"}

fish_dict=model.names
print("Model classes: ",fish_dict)

class_colors=dict(zip(fish_dict.keys(),np.linspace(0,255,len(fish_dict.keys())+1,dtype=int)[1:]))

# fish_dict={ 0: 'fish'}

num_classes=len(fish_dict.items())
print("num_classes is: ",num_classes)


def generate_IS_bwmask(mask,h,w):
    mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
    # cv2.imwrite("Maskara.jpg", mask_raw)
    return np.squeeze(mask_raw)

# First interval should be for background class!
class_colors=dict(zip(fish_dict.keys(),np.linspace(0,255,len(fish_dict.keys())+1,dtype=int)[1:]))


print("MASKS COLORS: ",class_colors)

out_path=os.path.join(data_path,"out")
if os.path.exists(out_path):
    shutil.rmtree(out_path)
    os.makedirs(out_path)

else:
    os.makedirs(out_path)


#GO THROUGH THE EXTRACTED IMGS DIR
for img in natsorted(os.listdir(data_path)):
    #just inference on left image
    if "left" in img and "mask" not in img and "yaml" not in img:
        print("running inference on ", img)
        print("------------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------------------------------------------------")
        img_path=os.path.join(data_path,img)
        results=model.predict(img_path,conf=0.05,project=out_path,name=masks_inf_folder,retina_masks=True,line_width=1,
                            boxes=True,show_labels=True,save=True,exist_ok=True,agnostic_nms=True,max_det=250)

        if(results[0].masks is not None):
            # fish_names=results[0].names
            fish_masks=results[0].masks
            fish_boxes=results[0].boxes
            h, w, c = results[0].orig_img.shape
            # masked=[np.ones((h,w))*255]*len(fish_dict.keys())
            masked = np.full( (len(fish_dict.keys()), h, w), 0)
            masked_dict=dict(zip(fish_dict.keys(),masked))
            mask_id=np.zeros((h,w))
            # print(masked_dict)
            num_masks=len(fish_masks)-1
            #Recorrer las mascaras inversamente porque va de menor a mayor confianza
            for i in range(num_masks+1):
                print("i is ",i, " num masks. ",num_masks)
                mask=fish_masks[num_masks-i]
                fish_cls=int(fish_boxes.cls[num_masks-i])
                fish_conf=float(fish_boxes.conf[num_masks-i])

                print("##############################################################################################")
                print("THIS FISH IS A: ",fish_dict[fish_cls],"with a confidence of : ",fish_conf)
                mask_bw=generate_IS_bwmask(mask,h,w)

                print("MASK BW SHAPE:",mask_bw.shape,"UNIQUE VALS:",np.unique(mask_bw))
                print("shape of masked_dict[fish_cls] ", masked_dict[fish_cls].shape)
                # máscara anterior (0s or ant masks) + el nuevo pez
                print("the colour is: ", class_colors[fish_cls] )
                print("the id is: ", i)

                inverted_mask=(mask_bw*(-1))+np.ones(mask_bw.shape)
                #Poner a 0 esos píxeles por si solapa con algún pez anterior
                mask_id=mask_id*inverted_mask
                #sumar la máscara
                mask_id=mask_id+mask_bw*i

                for key in fish_dict:
                    if key==fish_cls:
                        print("THE DETECTED CLASS")
                        #posar a 0 per les màscares que tenen overlap
                        masked_dict[key]=masked_dict[key]*inverted_mask
                        masked_dict[fish_cls]=masked_dict[fish_cls]+(mask_bw*class_colors[fish_cls])
                    else:
                        #debería poner las otras clases a 0 en esos pixeles por si solapan
                        #las confianzas están ordenadas de menor a mayor
                        cv2.imwrite("inverted_mask.png",inverted_mask*255)
                        masked_dict[key]=masked_dict[key]*inverted_mask


                print("CLASS Mask UNIQUE: ",np.unique(masked_dict[fish_cls]))
                print(masked_dict[fish_cls].shape)

            final_masks = np.stack(list(masked_dict.values()))

            # for mask in final_masks: THIS IS A CHECK
            #     print("Mask UNIQUE: ",np.unique(mask))

            mask_final=np.sum(final_masks,axis=0)
            print("FINAL MASK SHAPE: ",mask_final.shape)
            print("UNIQUE VALUES:",np.unique(mask_final))

            print("ID MASK SHAPE: ",mask_id.shape)
            print("ID MASK UNIQUE VALUES:",np.unique(mask_id))

            save_img_path=os.path.join(out_path,masks_inf_folder,img.split("_left")[0]+"_masked.png")
            save_idmask_path=os.path.join(out_path,masks_inf_folder,img.split("_left")[0]+"_class_ids.png")
            cv2.imwrite(save_img_path,mask_final)
            cv2.imwrite(save_idmask_path,mask_id)
            print("LA MASCARA FINAL!! ",np.unique(mask_final))
            print("Writing mask to: ",save_img_path)
            print("Writing mask with ids to: ",save_idmask_path)
            # Show the masked part of the image
