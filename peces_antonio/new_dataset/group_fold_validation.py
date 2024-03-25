import os
import pandas as pd
import json


# -------- argparser ---------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project_path", help="Folder containing fold trainings")
args = parser.parse_args()
project_path = args.project_path
# ----------------------------

# project_path = r"C:\Users\haddo\yolov8\peces_antonio\new_dataset\new_pipeline\kfold_large_1280_own_lr_0.005\large"

df = pd.DataFrame(columns=['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)'])

for fold_idx in range(1, 6):
    fold_validation_path = os.path.join(project_path, f"fold_{fold_idx}", "val", "validation_results.json")
    
    with open(fold_validation_path) as file:
        val_dict = json.load(file)
        
    df.loc[len(df)] = val_dict
        
df.to_csv(os.path.join(project_path, "fold_results.csv"), index=False)