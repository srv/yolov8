

from ultralytics import YOLO

data_path=r"C:\Users\Uib\yolov8\peces_antonio\16_classes_OD\dataset\test\images"

model_path=r"C:\Users\Uib\yolov8\peces_antonio\16_classes_OD\small\weights\best.pt"

#yolo predict model="C:\Users\Uib\yolov8\peces_antonio\nano\fold_3\weights\best.pt" source='C:\Users\Uib\yolov8\peces_antonio\seleccion_yolanda'
# Load a pretrained YOLOv8n model
model = YOLO(model_path)

# Run inference on 'bus.jpg' with arguments
model.predict(data_path, save=True, imgsz=640, conf=0.5)

# Run inference on the source
#results = model(data_path, stream=True)  # generator of Results objects
