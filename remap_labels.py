import os

# Specify the path to your labels folder
labels_folder = '/mnt/c/Users/haddo/yolov8/datasets/PLOME_IS_binary/test/labels'

# Function to remap class IDs in a single label file
def remap_class_ids(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.read().splitlines()

    modified_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            parts[0] = '0'  # Replace class ID with 0
        modified_lines.append(' '.join(parts))

    with open(output_file, 'w') as file:
        file.write('\n'.join(modified_lines))

# Process each file in the folder
for filename in os.listdir(labels_folder):
    if filename.endswith('.txt'):
        input_file = os.path.join(labels_folder, filename)
        output_file = os.path.join(labels_folder, 'remapped_' + filename)
        remap_class_ids(input_file, output_file)

print("Class IDs remapped for all label files.")