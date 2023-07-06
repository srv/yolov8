from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-seg.pt')  # load an official model

model = YOLO('/mnt/c/Users/haddo/yolov8/peixos/large/16_species/lr_0.0033/da/fold_4_seed_79/weights/best.pt',)  # pretrained YOLOv8n model

# Validate the model
metrics = model.val(data="/mnt/c/Users/haddo/yolov8/datasets/Instance_con_SAM_16/test_config.yaml")  # no arguments needed, dataset and settings remembered
print("DA METRICS YEAH")
print("box map 50-95:",metrics.box.map  )  # map50-95(B)
print("map50 B: ",metrics.box.map50)  # map50(B)
print("map 75 B: ",metrics.box.map75)  # map75(B)
print("B map 50-95 per category: ",metrics.box.maps )  # a list contains map50-95(B) of each category
print("mask map 50-95:",metrics.seg.map  )  # map50-95(M)
print("seg-map 50 masks",metrics.seg.map50)  # map50(M)
print("seg-map 75 masks",metrics.seg.map75)  # map75(M)
print("M map 50-95 per category: ",metrics.seg.maps )  # a list contains map50-95(M) of each category

print("metrics:",metrics.type)

box_map_50_95=[]
box_map_50=[]
box_map_75=[]


seg_map_50_95=[]
seg_map_50=[]
seg_map_75=[]