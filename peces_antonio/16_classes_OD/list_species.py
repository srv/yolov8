import os

dataset_path = r"C:\Users\Uib\yolov8\peces_antonio\16_classes_OD\dataset"

species = ['Chromis chromis', 'Coris julis', 'Dentex dentex', 'Diplodus annularis', 'Diplodus sargus', 'Diplodus vulgaris', 'Epinephelus marginatus', 'Lithognathus mormyrus', 'Mugilidae prob Chelon', 'Oblada melanura', 'Pomatous salator', 'Sciena umbra', 'Seriola dumerili', 'Serranus', 'Spicara maena', 'Spondyliosoma cantharus']

species_dict = {}
for idx, specie in enumerate(species): 
    species_dict[idx] = {
        'specie': specie, 
        'train_count': 0, 
        'valid_count': 0, 
        'test_count': 0, 
        'total': 0
    }

splits = ['train', 'valid', 'test'] 
for split in splits: 
    for filename in os.listdir(os.path.join(dataset_path, split, 'labels')): 
        with open(os.path.join(dataset_path, split, 'labels', filename), 'r') as file: 
            lines = file.readlines()
            for line in lines: 
                line = line.split()
                class_idx = int(line[0])
                species_dict[class_idx][f'{split}_count'] += 1
                species_dict[class_idx]['total'] += 1

with open(os.path.join(dataset_path, 'class_resume.txt'), 'a+') as resume_file:
    for idx in range(len(species)): 
        resume_file.write(str(species_dict[idx]['specie']) + '\n')
        resume_file.write('Train: ' + str(species_dict[idx]['train_count']) + '\n')
        resume_file.write('Valid: ' + str(species_dict[idx]['valid_count']) + '\n')
        resume_file.write('Test: ' + str(species_dict[idx]['test_count']) + '\n')
        resume_file.write('Total: ' + str(species_dict[idx]['total']) + '\n')   
        resume_file.write('-------------------------------------------\n')



    

