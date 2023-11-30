from ultralytics import YOLO

# Load a model


model = YOLO('yolov8m.pt')  # build from YAML and transfer weights

data_path="/home/uib/DATA/PEIXOS/PLOME_16ESP_OD/data.yaml"

# Train the model
results = model.train(data=data_path, epochs=100, imgsz=640,project="peixos/OD/",name="small_default")