import os
import json
import shutil
import numpy as np
import pandas as pd

raw_data_path = r"C:\Users\haddo\yolov8\peces_antonio\hyp_study\raw_val_data.json"
save_path = r"C:\Users\haddo\yolov8\peces_antonio\hyp_study\hyp_study_results.csv" # CSV extension pls

with open(raw_data_path, 'r') as file: 
    raw_data = json.load(file)
res = raw_data[""]

df = pd.DataFrame(columns=['model_version', 'F1(B)', 'F1(M)', 'mAP50(B)', 'mAP50(M)', 'mAP50-95(B)', 'mAP50-95(M)'])

for model_version in res.keys():
    current = res[model_version]

    row = [0 for _ in range(len(df.columns))]
    row[0] = model_version
    
    # F1 Score (B)
    row[1] = 2 * (current['metrics/recall(B)'] * current['metrics/precision(B)']) / (current['metrics/recall(B)'] + current['metrics/precision(B)'])

    # F1 Score (M)
    row[2] = 2 * (current['metrics/recall(M)'] * current['metrics/precision(M)']) / (current['metrics/recall(M)'] + current['metrics/precision(M)'])

    # mAP50
    row[3] = current['metrics/mAP50(B)']
    row[4] = current['metrics/mAP50(M)']

    # mAP50-95
    row[5] = current['metrics/mAP50-95(B)']
    row[6] = current['metrics/mAP50-95(M)']

    df.loc[len(df)] = row

df.to_csv(save_path)