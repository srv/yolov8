import os
import json
import pandas as pd

project_name = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\new_pipeline\kfold_test_all_sizes_haddock\nano"
k = 5

# Compute mean value (fold metrics)
keys = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']

# Results init
mean_data = {key:0 for key in keys}

for fold_idx in range(1, k+1):
    fold_data_path = os.path.join(project_name, f"fold_{fold_idx}", "val", "validation_results.json")
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
df.to_csv(os.path.join(project_name, "results.csv"), index=False)