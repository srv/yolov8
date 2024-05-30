from ultralytics import YOLO
import os
import json
import pandas as pd

# CAUTION, WE ARE ASSUMING THAT /folds/train/images AND /folds/valid/images
# CONTAIN THE IMAGES OF THE INTERESTED FOLD

# ONLY USE THIS JUST AFTER THE ERROR OCURRED AND BEFORE EXECUTING ANYTHING ELSE!!!

if __name__ == "__main__":
    imgsz = 1280
    device = "0"
    project_name = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\new_pipeline\kfold_large_1280_own_lr_0.01_cls_2.0\large"
    dataset_yaml = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\dataset\data_5_fold.yaml"

    for fold_idx in ("1", "2", "3", "4", "5"):

        val_dict = dict(
            data=r'{}'.format(dataset_yaml),
            imgsz=imgsz, 
            device=device, 
            split="test", 
            name="test"    
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

        with open(os.path.join(project_name, f"fold_{fold_idx}", "test", "test_results.json"), "w") as json_file:
            json.dump(results_data, json_file, indent=4)

    # Compute mean value (fold metrics)
    keys = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']

    # Results init
    mean_data = {key:0 for key in keys}
    k = 5
    for fold_idx in range(1, k+1):
        fold_data_path = os.path.join(project_name, f"fold_{fold_idx}", "test", "test_results.json")
        with open(fold_data_path, 'r') as file: 
            fold_data = json.load(file)
            
        for key in keys:
            mean_data[key] += fold_data[key]
        mean_data["F1(B)"] = 2 * (mean_data['metrics/recall(B)'] * mean_data['metrics/precision(B)']) / (mean_data['metrics/recall(B)'] + mean_data['metrics/precision(B)'])
        mean_data["F1(M)"] = 2 * (mean_data['metrics/recall(M)'] * mean_data['metrics/precision(M)']) / (mean_data['metrics/recall(M)'] + mean_data['metrics/precision(M)'])


    keys.append("F1(B)")
    keys.append("F1(M)")
    mean_data = {key: mean_data[key]/k for key in keys}

    df = pd.DataFrame.from_dict([mean_data], orient="columns")
    df.to_csv(os.path.join(project_name, "test_results.csv"), index=False)