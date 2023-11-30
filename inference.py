

from ultralytics import YOLO
import os

# data_path="/home/uib/DATA/PEIXOS/PLOME_16ESP_OD/test/images/"
data_path="/home/uib/PLOME/stereo_tests/test_stereo_andratx_interior/left/"

model_path="/home/uib/PLOME/fish_trained_models/yolov8/binary_fish/"
# model_path="/home/uib/PLOME/fish_trained_models/yolov8/16_classes"

out_path="/home/uib/PLOME/stereo_tests/test_stereo_andratx_interior/"

# model_name="yolov8lr_medium_16cIS_f2.pt"
model_name="yolov8lr_medium_BF_f2.pt.pt"

yolo_model=os.path.join(model_path,model_name)

#yolo predict model="C:\Users\Uib\yolov8\peces_antonio\nano\fold_3\weights\best.pt" source='C:\Users\Uib\yolov8\peces_antonio\seleccion_yolanda'
# Load a pretrained YOLOv8n model
model = YOLO(yolo_model)

# Run inference on 'bus.jpg' with arguments
model.predict(data_path, save=True, imgsz=1024, conf=0.4, \
    project=out_path,name="inference_yv8mbf",retina_masks=True,line_width=5,boxes=False,show_labels=True)

# Run inference on the source
#results = model(data_path, stream=True)  # generator of Results objects
