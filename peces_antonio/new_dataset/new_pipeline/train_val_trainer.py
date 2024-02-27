from ultralytics.utils import SETTINGS
SETTINGS['clearml'] = False

import os
from ultralytics import YOLO
import time 
from datetime import datetime, timedelta


if __name__ == "__main__":

    model_sizes = {
        # "n": "nano",
        # "s": "small",
        # "m": "medium",
        "l": "large",
        # "x": "extra_large"
    }

    yolo_name = "train_val_large_1280"

    project_path = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\new_pipeline"
    data_path = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\dataset\data.yaml"

    batch = 5
    seed = 42
    epochs = 300
    patience = 100
    imgsz = 1280

    for model_size, model_name in model_sizes.items():
            model = YOLO(f'yolov8{model_size}-seg.pt')

            model.train(
                data=data_path,
                epochs=epochs,
                patience=patience,
                batch=batch,
                project=project_path,
                name=os.path.join(yolo_name, model_name),
                seed=seed, 
                imgsz=imgsz
            )

            print('Toy durmiendo ._. zzZ')
            sleep_time = 5 #min
            print(f"Me despierto a las {(datetime.now() + timedelta(minutes=sleep_time)).strftime('%H:%M:%S')} :(")
            time.sleep(sleep_time*60)
