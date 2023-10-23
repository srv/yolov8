
import os
path_to_project = "/mnt/c/Users/Uib/yolov8/peixos/"
path_to_dataset_base = "/mnt/c/Users/haddo/yolov8/datasets/PLOME_IS_ANTONIO/"

model = "yolov8s-seg.pt"

config="/mnt/c/Users/haddo/yolov8/ultralytics/yolo/cfg/da.yaml"
data="/mnt/c/Users/haddo/yolov8/datasets/PLOME_IS_ANTONIO/data.yaml"

lr=0.003
project_name="/mnt/c/Users/haddo/yolov8/peixos/nano/antonio/"
run_name="small_adam_0.003"

train_instruction = "yolo segment train cfg={} data={} model={} epochs=200 imgsz=640 seed=42  lr0={} project={} name={}"
val_instruction = "yolo segment val data={} model={}  project={} name={} split={}"
valtest_instruction = "yolo segment val data={} model={}  project={} name={} split={}"

train_instruction_formatted=train_instruction.format(config,data,model,str(lr),project_name,run_name) 
val_instruction_formatted =val_instruction.format(data,os.path.join(project_name,run_name,"weights/"+"best.pt"),project_name,run_name+"/validation","val") 
valtest_instruction_formatted =val_instruction.format(data,os.path.join(project_name,run_name,"weights/"+"best.pt"),project_name,run_name+"/test","test")

os.system(train_instruction_formatted)
os.system(val_instruction_formatted)
os.system(valtest_instruction_formatted)

lr=0.001
run_name="small_adam_0.001"
train_instruction_formatted=train_instruction.format(config,data,model,str(lr),project_name,run_name) 
val_instruction_formatted =val_instruction.format(data,os.path.join(project_name,run_name,"weights/"+"best.pt"),project_name,run_name+"/validation","val") 
valtest_instruction_formatted =val_instruction.format(data,os.path.join(project_name,run_name,"weights/"+"best.pt"),project_name,run_name+"/test","test")

os.system(train_instruction_formatted)
os.system(val_instruction_formatted)
os.system(valtest_instruction_formatted)

train_instruction = "yolo segment train cfg=/mnt/c/Users/haddo/yolov8/ultralytics/yolo/cfg/dat_hsv_test.yml \
data=/mnt/c/Users/haddo/yolov8/datasets/PLOME_IS_ANTONIO/da_dataset.yaml model=yolov8n-seg.pt epochs=200 imgsz=640 seed=42 project=/mnt/c/Users/haddo/yolov8/peixos/nano/antonio/ name=da_test"