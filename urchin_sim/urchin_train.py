from ultralytics import YOLO
import os

project_path = r"/home/antonio/yolov8/urchin_sim"
data_path = os.path.join(project_path, "SimDataset/data.yaml")

model_size = 'n'
model = YOLO(f"yolov8{model_size}.pt")

model.train(
    data=data_path, 
    epochs=100, 
    patience=30, 
    batch=32,
    project=project_path, 
    name="SimDataset_train_1"
)

