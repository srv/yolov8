import cv2
import numpy as np
import random
from natsort import natsorted
import os
import shutil


def mask_to_bbox(mask_points):
    # Find the minimum and maximum coordinates
    min_x = min(point[0] for point in mask_points)
    min_y = min(point[1] for point in mask_points)
    max_x = max(point[0] for point in mask_points)
    max_y = max(point[1] for point in mask_points)

    # Create the bounding box as (x, y, x, y)
    bbox = (min_x, min_y, max_x, max_y)

    # # Calculate the width and height of the bounding box
    # width = max_x - min_x
    # height = max_y - min_y
    # # Create the bounding box as (x, y, w, h)
    # bbox = (min_x, max_x, width, height)

    return bbox

def overlay_segmentation_masks(image_path, label_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    # image_height, image_width, _ = image.shape
    image_height, image_width, _ = image.shape
    print("image_shape: height, width:,",image_height,image_width)

    # Read label file
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Create an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_image = image.copy()
    output_image=image.copy()
    boxes = []
    classes = []

    for line in lines:
        # Extract class index and bounding box information
        class_index = float(line.split()[0])
        classes.append(class_index)
        print(class_index)
        
        print("instance of class: ", class_index," : ",fish_dict[class_index])

        # Extract segmentation mask coordinates
        mask_coordinates = list(map(float, line.split()[1:]))

        # Reshape the coordinates into x-y pairs
        mask_coordinates = np.array(mask_coordinates).reshape(-1, 2)

        # Scale the mask coordinates based on image size
        mask_coordinates[:, 0] = mask_coordinates[:, 0] * image_width
        mask_coordinates[:, 1] = mask_coordinates[:, 1] * image_height

        if len(mask_coordinates>0):
            box = mask_to_bbox(mask_coordinates)
            boxes.append(box)
            
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            # Draw the mask using polygon
            cv2.fillPoly(mask, [mask_coordinates.astype(np.int32)], 255)

            # Generate a unique color for the class
            if class_index not in mask_colors:
                available_color=False
                while not available_color:
                    color= (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    if color not in mask_colors.values():
                        mask_colors[class_index] = color
                        available_color=True

            # Create a semi-transparent mask image
            mask_image[mask == 255] = mask_colors[class_index]

            # Overlay the mask on the original image with transparency
            alpha = 0.5  # Adjust the transparency level (0.0 to 1.0)
            output_image = cv2.addWeighted(mask_image, alpha, image, 1 - alpha, 0)

    for class_idx, box in zip(classes, boxes):
        # Plot bounding box
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        cv2.rectangle(output_image, p1, p2, color=mask_colors[class_idx], thickness=2)

        # Plot class name and background rectangle
        label = fish_dict[class_idx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25
        font_thickness = 1
        (text_width, text_height) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=font_thickness)[0]

        cv2.rectangle(
            output_image,
            (p1[0] - 1, p1[1] - text_height - 2), 
            (p1[0] + text_width + 1, p1[1]), 
            mask_colors[class_idx], 
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
    output_filename=output_path+"segmented_"+image_path.split("/")[-1]
    cv2.imwrite(output_filename, output_image)

    print("Saved segmented image:", output_filename)

    # Display the original image with all the segmentation masks overlayed
    # cv2.imshow('Segmentation Masks', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Provide the paths to the image and label files

images_path = '/home/antonio/yolov8/datasets/fish/PLOME_IS/train/images/'
labels_path = '/home/antonio/yolov8/datasets/fish/PLOME_IS/train/labels/'
output_path = '/home/antonio/Desktop/test/'
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

fish_species = [
    'Chromis chromis',
    'Coris julis', 
    'Dentex dentex', 
    'Diplodus annularis', 
    'Diplodus sargus', 
    'Diplodus vulgaris', 
    'Epinephelus marginatus', 
    'Lithognathus mormyrus', 
    'Mugilidae prob Chelon', 
    'Oblada melanura', 
    'Pomatous salator', 
    'Sciena umbra', 
    'Seriola dumerili', 
    'Serranus', 
    'Spicara maena', 
    'Spondyliosoma cantharus'
]

idxs=list(range(0,len(fish_species)))
mask_colors = {}
fish_dict=dict(zip(idxs,fish_species))
print(fish_dict)

images_list=natsorted(os.listdir(images_path))
labels_list=natsorted(os.listdir(labels_path))

for img,lbl in zip(images_list,labels_list):
    print("image is: ",img)
    print("lbl is: ",lbl)
    image_path=images_path+img
    label_path=labels_path+lbl
    if img.split(".")[-2] ==lbl.split(".")[-2]:
        # Call the function to overlay the segmentation masks on the image and save them
        overlay_segmentation_masks(image_path, label_path, output_path)
    else:
        print("something is wrong from here!!!!")
        print("img is: ",img)
        print("lbl is: ",lbl)