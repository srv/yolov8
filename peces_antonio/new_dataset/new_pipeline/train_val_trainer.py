from ultralytics.utils import SETTINGS
SETTINGS['clearml'] = False

import os
import torch
from ultralytics import YOLO
import time 
from datetime import datetime, timedelta
import json


if __name__ == "__main__":

    model_sizes = {
        # "n": "nano",
        "s": "small",
        # "m": "medium",
        "l": "large",
        # "x": "extra_large"
    }

    project_path = r"D:\yolov8\peces_antonio\new_dataset\new_pipeline"
    data_path = r"D:\yolov8\peces_antonio\new_dataset\dataset\data.yaml"
    cfg_path = r"D:\yolov8\peces_antonio\configs\best_da_cls2.0.yaml"
    yolo_name = f"train_val_1280"

    batch = 7
    seed = 42
    epochs = 300
    patience = 100
    imgsz = 1280
    optimizer = 'SGD'
    lr0 = 0.01

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
            imgsz=imgsz, 
            cfg = cfg_path, 
            optimizer=optimizer, 
            lr0 = lr0, 
            workers=4
        )
        del model
        torch.cuda.empty_cache()


        print('Toy durmiendo ._. zzZ')
        sleep_time = 5 #min
        print(f"Me despierto a las {(datetime.now() + timedelta(minutes=sleep_time)).strftime('%H:%M:%S')} :(")
        time.sleep(sleep_time*60)

        val_dict = dict(
        data=data_path,
        imgsz=imgsz, 
        split="val", 
        name="val"    
        )


        model = YOLO(os.path.join(project_path, yolo_name, model_name, "weights", "best.pt"))

        val_dict["project"] = os.path.join(project_path, yolo_name, model_name)

        results = model.val(**val_dict)
        del model


        results_data = {}
        # Keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']
        for key, value in zip(results.keys, results.mean_results()):
            results_data[key] = value

        #TODO: Store also fitness values (?)

        print(json.dumps(results_data, indent=4))

        with open(os.path.join(project_path, yolo_name, model_name, "val", "validation_results.json"), "w") as json_file:
            json.dump(results_data, json_file, indent=4)

        torch.cuda.empty_cache()



