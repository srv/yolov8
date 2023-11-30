from ultralytics import YOLO
import os
import cv2
import numpy as np
# Load a model
import matplotlib.pyplot as plt

model = YOLO('/home/uib/yolov8/trained_models/tl_nano_16/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images

exemple_imgs=os.listdir("inference_fish/")
img="inference_fish/left_frame000037.png"

results = model(img)

if(results[0].masks is not None):
    # Convert mask to single channel image
    # black_img=np.zeros((results[0].orig_img.shape))
    masked=np.ones((768,1024))*255
    i=0
    for mask in results[0].masks:

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
        cv2.imwrite("mask_"+str(i)+".jpg", masked)
        # Invert the mask to get everything but black
        i+=1

    mask_final = masked*(-1)+255
    cv2.imwrite("mask_"+str(i)+".jpg", mask_final)
        # Show the masked part of the image



# if(results[0].masks is not None):
#     # Convert mask to single channel image
#     mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)

#     # Convert single channel grayscale to 3 channel image
#     mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

#     # Get the size of the original image (height, width, channels)
#     h2, w2, c2 = results[0].orig_img.shape

#     # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
#     mask = cv2.resize(mask_3channel, (w2, h2))

#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

#     # Define range of brightness in HSV
#     lower_black = np.array([0,0,0])
#     upper_black = np.array([0,0,1])

#     # Create a mask. Threshold the HSV image to get everything black
#     mask = cv2.inRange(mask, lower_black, upper_black)

#     # Invert the mask to get everything but black
#     mask = cv2.bitwise_not(mask)

#     # Apply the mask to the original image
#     masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)

#     # Show the masked part of the image
#     cv2.imwrite("mask.jpg", mask)
#     # cv2.imwrite(output_dir+str(name_img)+'.jpg',frameL)