from ultralytics import YOLO
import os
def main():
    project_path = r"C:\Users\Uib\yolov8\urchin_sim"
    data_path = os.path.join(project_path, "SimDataset/data.yaml")

    model_sizes = ["n", "s", "m", "l", "x"]
    for model_size in model_sizes:
        model = YOLO(f"yolov8{model_size}.pt")

        model.train(
            data=data_path, 
            epochs=300, 
            patience=20, 
            batch=32,
            project=project_path, 
            name=f"train_{model_size}"
        )

if __name__ == "__main__":
    main()
