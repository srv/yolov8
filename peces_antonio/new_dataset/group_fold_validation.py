import os
import pandas as pd
import json

project_path = r"..."
model_sizes = ["large"]

df = pd.DataFrame(columns=['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)'])

for model_size in model_sizes:
    for fold_idx in range(1, 6):
        fold_validation_path = os.path.join(project_path, model_size, f"fold_{fold_idx}", "val", "validation_results.json")
        
        with open(fold_validation_path) as file:
            val_dict = json.load(file)
            print(val_dict)
            
        df.loc[len(df)] = val_dict
            
    df.to_csv(os.path.join(project_path, model_size, "fold_results.csv"))