from ultralytics import YOLO
import os

project_path = r'C:\Users\Uib\yolov8\peces_antonio\16_classes_OD\small'
data = r'C:\Users\Uib\yolov8\peces_antonio\16_classes_OD\dataset\data.yaml'
model_path = os.path.join(project_path, 'weights', 'best.pt')



if __name__ == '__main__': 

    model = YOLO(model_path)

    model.val(project=project_path, name='test', data=data, split='test')