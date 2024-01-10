import os

study_path = f"/home/antonio/yolov8/peces_antonio/hyp_study"

for folder in os.listdir(study_path): 
    path = os.path.join(study_path, folder)
    if os.path.isdir(path):
        contained = os.listdir(path)
        if "results.csv" in contained:
            print(path)

            instruction = f"python /home/antonio/fish_utils/antonio_utils/plot_train_results.py --results_csv_path {os.path.join(path, 'results.csv')} --save_path {path}"
            
            os.system(instruction)