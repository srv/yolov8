from ultralytics import YOLO

model_path = r"D:\yolov8\peces_antonio\new_dataset\new_pipeline\kfold_large_1280_own_lr_0.001\large\fold_4\weights\last.pt"

if __name__ == "__main__": 
    model = YOLO(model_path)
    model.train(resume=True)