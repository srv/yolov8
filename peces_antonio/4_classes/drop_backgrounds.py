import os
import shutil
import random
import numpy as np

RANDOM_SEED = 42
BACKGROUND_RATIO = 0.1

# Dataset with all available classes
full_dataset = r'C:\Users\Uib\yolov8\peces_antonio\dataset'

# Dataset with reduced number of classes
dataset_path = r'C:\Users\Uib\yolov8\peces_antonio\4_classes\nb\dataset_nb'

folders = ['train', 'valid']

for folder in folders: 
    images_path = os.path.join(dataset_path, folder, 'images')
    labels_path = os.path.join(dataset_path, folder, 'labels')
    
    deleted = []        

    for img in os.listdir(images_path):
        lbl = f"{'.'.join(img.split('.')[:-1])}.txt"

        # Check if img is background in reduced-class dataset
        background = False 
        if os.path.exists(os.path.join(labels_path, lbl)):
            with open(os.path.join(labels_path, lbl), 'r') as file: 
                lines = file.readlines()
                if lines == []:
                    background = True

        else: 
            background = True
        
        # if image has no labels but is not a real background, it means that unclassified fish appears in the image -> delete!
        if background: 
            deleted.append(img)
            os.remove(os.path.join(images_path, img))
            if os.path.exists(os.path.join(labels_path, lbl)): 
                os.remove(os.path.join(labels_path, lbl))

        with open(os.path.join(dataset_path, 'removed_unlabeled_images.txt'), 'a') as file:
            for img in deleted:
                file.write(f'{img}\n')

    n_images = len(os.listdir(images_path))


    images_path = os.path.join(full_dataset, folder, 'images')
    labels_path = os.path.join(full_dataset, folder, 'labels')
    
    real_backgrounds = []
    for img in os.listdir(images_path): 
        lbl = f"{'.'.join(img.split('.')[:-1])}.txt"

        # Check real backgrounds using dataset with 16 classes
        if os.path.exists(os.path.join(full_dataset, folder, 'labels', lbl)):
            with open(os.path.join(full_dataset, folder, 'labels', lbl), 'r') as file: 
                lines = file.readlines()
                if lines == []: 
                    real_backgrounds.append(img)
        else: 
            real_backgrounds.append(img)

        

    n_background = len(real_backgrounds)
    n_final_background = np.round( (n_images) / (1 / BACKGROUND_RATIO - 1))

    print(f'{folder} folder: ')
    print(f'-> {n_images} labeled images (at least one label).')
    print(f'-> {n_background} real backgrounds in 16-class dataset.')
    print(f'-> With {BACKGROUND_RATIO} background ratio, {n_final_background} needed.')

    if n_background - n_final_background < 0: 
        print(f'No sufficient background files to accomplish {BACKGROUND_RATIO} background ratio! Using all of them {n_background}.')
    else: 
        print(f'Dropping {n_background - n_final_background} background images randomly!')
    print()
    

    random.seed(RANDOM_SEED)
    random.shuffle(real_backgrounds)

    for idx, img in enumerate(real_backgrounds[:int(n_final_background)]):
        lbl = f"{'.'.join(img.split('.')[:-1])}.txt"

        shutil.copy(os.path.join(full_dataset, folder, 'images', img), os.path.join(dataset_path, folder, 'images', img))
        with open(os.path.join(dataset_path, folder, 'labels', lbl), 'w') as file: 
            pass
            
    with open(os.path.join(dataset_path, 'added_backgrounds.txt'), 'a') as file:
        for img in real_backgrounds[:int(n_final_background)]:
            file.write(f'{img}\n')