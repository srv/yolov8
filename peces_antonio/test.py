from ultralytics import YOLO

def main():
    project_path = r"C:\Users\haddo\yolov8\peces_antonio\fold2_retrain"
    data_path = r"C:\Users\haddo\yolov8\peces_antonio\dataset\data_fold_2.yaml"
    best_da_cfg = r"C:\Users\haddo\yolov8\peces_antonio\hyp_study\best_da_config.yaml"


    model = YOLO('yolov8x-seg.pt')
    model.train(
        data = data_path, 
        epochs = 500, 
        batch = 8, 
        seed = 42, 
        lr0 = 0.01, 
        patience = 0, 
        optimizer = 'SGD', 
        cfg = best_da_cfg,
        project = project_path, 
        name = 'fold_2'
    )

    model.val(data=data_path)


if __name__ == "__main__":
    main()