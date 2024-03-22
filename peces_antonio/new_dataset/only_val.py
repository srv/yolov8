from ultralytics import YOLO
import os
import json

# CAUTION, WE ARE ASSUMING THAT /folds/train/images AND /folds/valid/images
# CONTAIN THE IMAGES OF THE INTERESTED FOLD

# ONLY USE THIS JUST AFTER THE ERROR OCURRED AND BEFORE EXECUTING ANYTHING ELSE!!!

if __name__ == "__main__":
    imgsz = 1280
    batch = 7
    device = "0"
    project_name = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\new_pipeline\kfold_large_1280_own_lr_0.005\large"
    fold_idx = "4"
    dataset_yaml = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\dataset\data_5_fold.yaml"

    val_dict = dict(
        data=r'{}'.format(dataset_yaml),
        imgsz=imgsz, 
        batch=batch, 
        device=device, 
        split="val", 
        name="val"    
    )


    model = YOLO(os.path.join(project_name, f"fold_{fold_idx}", "weights", "best.pt"))

    val_dict["project"] = os.path.join(project_name, f"fold_{fold_idx}")

    results = model.val(**val_dict)


    results_data = {}
    # Keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']
    for key, value in zip(results.keys, results.mean_results()):
        results_data[key] = value

    #TODO: Store also fitness values (?)

    print(json.dumps(results_data, indent=4))

    with open(os.path.join(project_name, f"fold_{fold_idx}", "val", "validation_results.json"), "w") as json_file:
        json.dump(results_data, json_file, indent=4)