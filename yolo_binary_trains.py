import os
import time

path_to_project = "/mnt/c/Users/haddo/yolov8/peixos/fish_no_fish/"
path_to_dataset = "/mnt/c/Users/haddo/yolov8/datasets/PLOME_IS_binary/"
dataset_yaml = path_to_dataset +  "/data.yaml"

model_sizes = {"yolov8n-seg.pt": "nano", "yolov8x-seg.pt": "xl"}
train_instruction = "yolo segment train data={} model={} epochs=200 imgsz=640 seed=42 project={} name={} seed=42 --patience 20"

val_instruction = "yolo segment val data={} model={}  project={} name={} split=val"

for model_size in model_sizes.keys():

    run_name = model_sizes[model_size]

    train_instruction_formatted=train_instruction.format(dataset_yaml,model_size,path_to_project,run_name) 

    val_instruction_formatted =val_instruction.format(dataset_yaml,os.path.join(path_to_project,run_name,"weights/"+"best.pt"),path_to_project,run_name+"/validation") 
    # test_instruction_formatted =test_instruction.format(dataset_yaml,os.path.join(project_name,run_name,"weights/"+"best.pt"),project_name,run_name+"/test") 
    
    with open('/mnt/c/Users/haddo/yolov8/calls_plome_binary.txt', 'a+') as f:
        f.write(train_instruction_formatted)
        f.write("\n")
        f.write(val_instruction_formatted)
        f.write("\n")
        # f.write(test_instruction_formatted)
        f.write("------------------------------------------------------------- \n")
        f.write("\n")
    
    os.system(train_instruction_formatted)
    os.system(val_instruction_formatted)
    
    time.sleep(300)
