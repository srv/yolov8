import os
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

project_path = r"/home/slimbook/yolov8/peces_antonio/new_dataset/new_pipeline/kfold_large_1280_own_lr_0.01_cls8.0_good/large"
dataset_path = r"/home/slimbook/yolov8/peces_antonio/new_dataset/dataset"
dataset_yaml = os.path.join(dataset_path, "data.yaml")
matrix_dir = os.path.join(project_path, "matrix_plots")

# ONLY USE IN TEST!!! TO USE IT IN VAL, MODIFICATIONS HAVE TO BE MADE IN ORDER TO PERFORM VALIDATION IN EACH FOLD!!!
split = "test"

mean_matrix = None

for fold_idx in range(1, 6):
    os.makedirs(os.path.join(matrix_dir, f"fold_{fold_idx}"), exist_ok=True)
    
    model = YOLO(os.path.join(project_path, f"fold_{fold_idx}", "weights", "best.pt"))

    # names = tuple(model.names.values())
    names = tuple(['Chromis chromis', 'Coris julis', 'Dentex dentex', 'D. annularis', 'D. sargus', 'D. vulgaris', 'E. marginatus', 'L. mormyrus', 'Mugilidae', 'O. melanura', 'P. salator', 'S. umbra', 'S. dumerili', 'S. cabrilla', 'S. scriba', 'S. maena', 'S. cantharus'])

    val_results = model.val(
        data=dataset_yaml,
        split=split,
        imgsz=1280, 
        batch=1,
        project = os.path.join(project_path, f"fold_{fold_idx}"),
        name = f"avg_matrix_validation", 
        exist_ok = True
    )
    
    matrix = val_results.confusion_matrix
    matrix.plot(save_dir = os.path.join(matrix_dir, f"fold_{fold_idx}"), normalize=False, names = names)
    matrix.plot(save_dir = os.path.join(matrix_dir, f"fold_{fold_idx}"), normalize=True, names = names)
    
    if fold_idx == 1:
        mean_matrix = matrix.matrix.copy()
    else:
        mean_matrix += matrix.matrix
    
mean_matrix /= 5

mean_matrix_obj = ConfusionMatrix(
    nc = matrix.nc
)
mean_matrix_obj.matrix = mean_matrix
mean_matrix_obj.plot(save_dir = matrix_dir, normalize=False, names = names)
mean_matrix_obj.plot(save_dir = matrix_dir, normalize=True, names = names)

    # model.predict(
    #     source = r"/home/slimbook/dataset_lanty/extracted_dataset/test/images", 
    #     batch = 1, 
    #     project = "./predict",
    #     name = f"fold_{fold_idx}", 
    #     save = True, 
    #     save_txt = True
    # )