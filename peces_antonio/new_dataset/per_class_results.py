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
    project_name = r"/home/azken/antonio/yolov8/peces_antonio/new_dataset/new_pipeline/kfold_large_1280_own_lr_0.01_cls8.0/large"
    dataset_path = r"/home/azken/antonio/yolov8/peces_antonio/new_dataset/dataset"
    dataset_yaml = os.path.join(dataset_path, "data.yaml")

    name = split

    # Initialize a dictionary to store metrics for each class
    class_metrics = {}

    for fold_idx in range(1, 6):
        # Generate temporal yaml for fold validation.
        with open(dataset_yaml, 'r') as file:
            data = yaml.safe_load(file)
            
        # Modify the parameters
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
        metrics_keys = list(results.results_dict.keys())
        metrics_keys.remove("fitness")

        # Get metrics for each class avoiding the fish class
        for class_idx, class_name in model.names.items():
           
            class_results = results.class_result(class_idx)
            
            if class_name not in class_metrics:
                class_metrics[class_name] = {key: [] for key in metrics_keys}

            for idx, metric in enumerate(metrics_keys):
                class_metrics[class_name][metric].append(class_results[idx])

    # Calculate the mean metrics for each class
    mean_class_metrics = {}
    for class_name, metrics in class_metrics.items():
        mean_class_metrics[class_name] = {key: sum(values) / len(values) for key, values in metrics.items()}

    # Convert the metrics dictionary to a DataFrame and save to CSV
    df = pd.DataFrame.from_dict(mean_class_metrics, orient='index')
    df.to_csv(os.path.join(project_name, f"{split}_resume_matrix_per_class.csv"), index_label='class')

    # Optionally, print the DataFrame
    print(df)
