# import os
# os.environ['YOLO_VERBOSE'] = 'false'

from ultralytics import YOLO
from ultralytics.utils import SETTINGS
SETTINGS['clearml'] = False

import argparse

'''
# -------------------------- Default config -------------------------------------
python train_yolov8.py --project_name "Peces" --task_name "Argparse test" \
--model_size "n" --dataset "/mnt/c/Users/Uib/Documents/peces/dataset/data.yaml" \
--optimizer "SGD" --epochs 200 --batch 8 --patience 20 \
--yolo_proj "./train_argparse_test/" --yolo_name "test"

# --------------------------- Custom config -------------------------------------
python train_yolov8.py --project_name "Peces" --task_name "Argparse test" \
--model_size "n" --config "./config.yaml" --dataset "./dataset/data.yaml" \
--optimizer "SGD" --epochs 200 --batch 8 --patience 20 \
--yolo_proj "./train_argparse_test/" --yolo_name "test"
'''

parser = argparse.ArgumentParser()
parser.add_argument('--model_size', help='YOLOv8 Model Size', 
                    choices=['n', 's', 'm', 'l', 'x'], default=None)
parser.add_argument('--pre_trained', help='Wheter to use YOLO pretrained weights or not',
                    default=True, type=bool)
parser.add_argument('--config', help='config.yaml path', default=None)
parser.add_argument('--dataset', help='dataset.yaml path')
parser.add_argument('--epochs', help='Number of epochs to train the model', type=int)
parser.add_argument('--optimizer', help='Optimizer to use during training', default=None)
parser.add_argument('--batch', help='Batch size during training', type=int)
parser.add_argument('--patience', help='Patience during training', default=None, type=int)
parser.add_argument('--yolo_proj', help='YOLOv8 project name where train will be saved')
parser.add_argument('--yolo_name', help='YOLOv8 save folder name')
parser.add_argument('--lr', help='Initial learning rate', type=float, default=None)
parser.add_argument('--seed', help='Seed during training', type=int, default=42)
parser.add_argument('--imgsz', help='Image size during training', type=int, default=640)
parser.add_argument('--val', help='Perform validation during training', type=bool, default=True)

args = parser.parse_args()

model_size = args.model_size
pretrained = args.pre_trained
config = args.config
dataset = args.dataset
epochs = args.epochs
optimizer = args.optimizer
batch = args.batch
patience = args.patience
yolo_proj = args.yolo_proj
yolo_name = args.yolo_name
lr = args.lr
seed = args.seed
imgsz = args.imgsz
val = args.val



def main():


    # model_size = 'x'
    model_variant = f'yolov8{model_size}-seg'
    # model_variant = f'yolov8{model_size}'

    if pretrained: 
        model = YOLO(f'{model_variant}.pt')
    else:
        model = YOLO(f'{model_variant}.yaml')


    train_args =  dict(
        data=dataset,
        epochs=epochs,
        patience=patience,
        batch=batch,
        project=yolo_proj,
        name=yolo_name,
        seed=seed, 
        imgsz=imgsz,
        val=val
    )

    if optimizer is not None: 
        train_args['optimizer'] = optimizer
    if config is not None:
        train_args['cfg'] = config
    if patience is not None:
        train_args['patience'] = patience
    if lr is not None: 
        train_args['lr0'] = lr

    result = model.train(**train_args)

    # model.val(project=f'{yolo_proj}/{yolo_name}', name='val')



if __name__=='__main__':
    main()