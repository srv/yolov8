

from ultralytics import YOLO

# data_path=r"C:\Users\Uib\yolov8\peces_antonio\16_classes_OD\dataset\test\images"

# model_path=r"C:\Users\Uib\yolov8\peces_antonio\16_classes_OD\small\weights\best.pt"

# #yolo predict model="C:\Users\Uib\yolov8\peces_antonio\nano\fold_3\weights\best.pt" source='C:\Users\Uib\yolov8\peces_antonio\seleccion_yolanda'
# # Load a pretrained YOLOv8n model
# model = YOLO(model_path)

# # Run inference on 'bus.jpg' with arguments
# model.predict(data_path, save=True, imgsz=640, conf=0.5)

# Run inference on the source
#results = model(data_path, stream=True)  # generator of Results objects



# data_path_original="/home/uib/PLOME/SARMIENTO/Original"
# data_path_processed="/home/uib/PLOME/SARMIENTO/Processed"
data_path="/home/uib/PLOME/SARMIENTO/fotos_lander_UPC"
model_path="/home/uib/yolov8/trained_models/binary_fish_detector_XL.pt"
# Load a pretrained YOLOv8n model
model = YOLO(model_path)


model.predict(data_path, save=True, imgsz=640, conf=0.2,project="/home/uib/PLOME/SARMIENTO/fotos_lander_UPC",name="infered")
# model.predict(data_path_processed, save=True, imgsz=640, conf=0.5,project="/home/uib/PLOME/SARMIENTO/infered",name="processed")


#EVALUATE:
# model_yaml="/home/uib/DATA/PEIXOS/PLOME IS.v15-binary.yolov8/data.yaml"
# model_path="/home/uib/yolov8/trained_models/fish_model_nano.pt"

# model = YOLO(model_path)
# # Validate the model
# metrics = model.val(data=model_yaml,imgsz=1024,split="test",plots=True)  # no arguments needed, dataset and settings remembered
# print(metrics.box.map)    # map50-95
# print(metrics.box.map50)  # map50
# print(metrics.box.map75)  # map75
# print(metrics.box.maps)   # a list contains map50-95 of each category