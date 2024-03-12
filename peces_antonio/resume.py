from ultralytics import YOLO

model_path = r"D:\yolov8\peces_antonio\new_dataset\new_pipeline\kfold_large_1280_noda\large\lr_0\fold_3\weights\last.pt"

if __name__ == "__main__": 
    model = YOLO(model_path)
    model.train(resume=True)