import cv2
import numpy as np
import random
from natsort import natsorted
import os
import shutil
import os

def compute_box(row, image_shape):
    center_x, center_y, width, height = row[0], row[1], row[2], row[3]
    x1 = int((center_x - width / 2) * image_shape[1])
    y1 = int((center_y - height / 2) * image_shape[0])
    x2 = int((center_x + width / 2) * image_shape[1])
    y2 = int((center_y + height / 2) * image_shape[0])
    return [x1, y1, x2, y2]

def overlay_segmentation_masks(image_path, label_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    # image_height, image_width, _ = image.shape
    image_height, image_width, _ = image.shape
    print("image_shape: height, width:,",image_height,image_width)

    # Read label file
    with open(label_path, 'r') as file:
        lines = file.readlines()

    output_image=image.copy()
    boxes = []
    classes = []

    for line in lines:
        line = list(map(float, line.split()))
        # Extract class index and bounding box information
        class_index = line[0]
        classes.append(class_index)
        print(class_index)

        print("instance of class: ", class_index," : ",fish_dict[class_index])

        box = compute_box(line[1:5], image_shape=(image_height, image_width))
        print(box)
        boxes.append(box)

        if class_index not in box_colors:
                available_color=False
                while not available_color:
                    color= (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    if color not in box_colors.values():
                        box_colors[class_index] = color
                        available_color=True

    for class_idx, box in zip(classes, boxes):
        # Plot bounding box
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        cv2.rectangle(output_image, p1, p2, color=box_colors[class_idx], thickness=6)

        # Plot class name and background rectangle
        label = fish_dict[class_idx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=font_thickness)[0]

        cv2.rectangle(
            output_image,
            (p1[0] - 1, p1[1] - text_height - 2),
            (p1[0] + text_width + 1, p1[1]),
            box_colors[class_idx],
            cv2.FILLED
        )
        cv2.putText(
            output_image,
            label,
            (p1[0], p1[1] - text_height//2),
            font,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=font_thickness
        )

    # Save the output image
    output_filename = os.path.join(output_path, image_path.split("/")[-1])
    cv2.imwrite(output_filename, output_image)

    print("Saved segmented image:", output_filename)

    # Display the original image with all the segmentation masks overlayed
    # cv2.imshow('Segmentation Masks', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Provide the paths to the image and label files

images_path = '/home/uib/DATA/PEIXOS/PLOME_16ESP_OD/test/images/'
labels_path = '/home/uib/DATA/PEIXOS/PLOME_16ESP_OD/test/labels/'
output_path = '/home/uib/DATA/PEIXOS/PLOME_16ESP_OD/test/painted_labels/'


if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

# fish_species= ['Chromis chromis', 'Coris julis', 'Dentex dentex', 'Diplodus annularis', 'Diplodus sargus', 'Diplodus vulgaris', 'Epinephelus marginatus', 'Lithognathus mormyrus', 'Mugilidae prob Chelon', 'Oblada melanura', 'Pomatous salator', 'Sciena umbra', 'Seriola dumerili', 'Serranus', 'Spicara maena', 'Spondyliosoma cantharus']
fish_species= ['Chromis chromis', 'Coris julis', 'Dentex dentex', 'Diplodus annularis', 'Diplodus sargus', 'Diplodus vulgaris', 'Epinephelus marginatus', 'Lithognathus mormyrus', 'Mugilidae prob Chelon', 'Oblada melanura', 'Pomatous salator', 'Sciena umbra', 'Seriola dumerili', 'Serranus', 'Spicara maena', 'Spondyliosoma cantharus']


idxs=list(range(0,len(fish_species)))
box_colors = {}
fish_dict=dict(zip(idxs,fish_species))
print(fish_dict)

images_list=natsorted(os.listdir(images_path))
labels_list=natsorted(os.listdir(labels_path))
print("IMAGES",images_list)

for img,lbl in zip(images_list,labels_list):
    print("image is: ",img)
    print("lbl is: ",lbl)
    image_path=os.path.join(images_path,img)
    label_path=os.path.join(labels_path,lbl)
    if img.split(".")[0] ==lbl.split(".")[0]:
        # Call the function to overlay the segmentation masks on the image and save them
        overlay_segmentation_masks(image_path, label_path, output_path)
    else:
        print("something is wrong from here!!!!")
        print("img is: ",img)
        print("lbl is: ",lbl)