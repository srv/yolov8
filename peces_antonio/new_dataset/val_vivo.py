from ultralytics import YOLO
import os

if __name__ == "__main__":

    nonvivo_project = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\new_pipeline\kfold_large_1280_own_lr_0.01\large"

    for fold_idx in range(1, 6):
        fold_path = os.path.join(nonvivo_project,  "lr_0.01", f"fold_{fold_idx}")
        model = YOLO(os.path.join(fold_path, "weights", "best.pt"))

        model.val(
            data = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\vivo_dataset\data.yaml",
            imgsz = 1280,
            batch=7,
            save=True,
            project=fold_path, 
            name="vivo_val",
            exist_ok=True)