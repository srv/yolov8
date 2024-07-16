from ultralytics import YOLO
import os
import json
import yaml
import pandas as pd

if __name__ == "__main__":
    split = "val"
    imgsz = 1280
    batch = 1
    device = "0"
    project_name = r"/home/slimbook/yolov8/peces_antonio/new_dataset/new_pipeline/kfold_large_1280_own_lr_0.01_cls8.0_good/large"
    dataset_path = r"/home/slimbook/yolov8/peces_antonio/new_dataset/dataset"
    dataset_yaml = os.path.join(dataset_path, "data.yaml")

    name = split

    for fold_idx in range(1, 6):            
        # Generate temporal yaml for fold validation.
        with open(dataset_yaml, 'r') as file:
            data = yaml.safe_load(file)
            
        # Modify the parameters
        if split == "val":
            data['train'] = "./"
            data['val'] = os.path.join(dataset_path, "folds", str(fold_idx), "images")
        
        # Save the modified file to another path
        fold_dataset_yaml = os.path.join(dataset_path, f"{fold_idx}_data.yaml")
        with open(fold_dataset_yaml, 'w') as file:
            yaml.safe_dump(data, file)

        val_dict = dict(
            data=fold_dataset_yaml,
            imgsz=imgsz, 
            batch=batch, 
            device=device, 
            split=split, 
            name=name, 
            exist_ok=True  
        )

        model = YOLO(os.path.join(project_name, f"fold_{fold_idx}", "weights", "best.pt"))

        val_dict["project"] = os.path.join(project_name, f"fold_{fold_idx}")

        results = model.val(**val_dict)
        
        results_data = {}
        # Keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']
        for key, value in zip(results.keys, results.mean_results()):
            results_data[key] = value

        print(json.dumps(results_data, indent=4))

        with open(os.path.join(project_name, f"fold_{fold_idx}", "val", f"batch_{batch}_validation_results.json"), "w") as json_file:
            json.dump(results_data, json_file, indent=4)
            
    
    # Compute mean value (fold metrics)
    keys = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']

    # Results init
    mean_data = {key:0 for key in keys}

    for fold_idx in range(1, 6):
        fold_data_path = os.path.join(project_name, f"fold_{fold_idx}", "val", f"batch_{batch}_validation_results.json")
        with open(fold_data_path, 'r') as file: 
            fold_data = json.load(file)
            
        for key in keys:
            mean_data[key] += fold_data[key]
        mean_data["F1(B)"] = 2 * (mean_data['metrics/recall(B)'] * mean_data['metrics/precision(B)']) / (mean_data['metrics/recall(B)'] + mean_data['metrics/precision(B)'])
        mean_data["F1(M)"] = 2 * (mean_data['metrics/recall(M)'] * mean_data['metrics/precision(M)']) / (mean_data['metrics/recall(M)'] + mean_data['metrics/precision(M)'])


    keys.append("F1(B)")
    keys.append("F1(M)")
    mean_data = {key: mean_data[key]/5 for key in keys}

    df = pd.DataFrame.from_dict([mean_data], orient="columns")
    df.to_csv(os.path.join(project_name, f"batch_{batch}_mean_val_results.csv"), index=False)