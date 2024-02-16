from clearml import Task, Logger
from ultralytics import YOLO

import argparse

'''
# -------------------------- Default config -------------------------------------
python clearml_log_yolov8.py --project_name "Peces" --task_name "Argparse test" \
--model_size "n" --dataset "/mnt/c/Users/Uib/Documents/peces/dataset/data.yaml" \
--optimizer "SGD" --epochs 200 --batch 8 --patience 20 \
--yolo_proj "./train_argparse_test/" --yolo_name "test"

# --------------------------- Custom config -------------------------------------
python clearml_log_yolov8.py --project_name "Peces" --task_name "Argparse test" \
--model_size "n" --config "./config.yaml" --dataset "./dataset/data.yaml" \
--optimizer "SGD" --epochs 200 --batch 8 --patience 20 \
--yolo_proj "./train_argparse_test/" --yolo_name "test"
'''

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', help='ClearML Project Name')
parser.add_argument('--task_name', help='ClearML Task Name')
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

args = parser.parse_args()

project_name = args.project_name
task_name = args.task_name
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


def on_fit_epoch_end(trainer):
    # Log loss data to ClearML
    for loss_type in ('box_loss', 'cls_loss', 'dfl_loss', 'seg_loss'):
        for split, dict in zip(('train', 'val'), (trainer.label_loss_items(trainer.tloss), trainer.metrics)):
            key = f'{split}/{loss_type}'
            if key in dict.keys():
                Logger.current_logger().report_scalar(
                    f'{loss_type}',
                    key,
                    iteration=trainer.epoch,
                    value=dict[key]
                )

    for key in trainer.metrics.keys(): 
        if 'metrics' in key:
            metric_type = (key.split('/')[1]).split('(')[0]
            Logger.current_logger().report_scalar(
                metric_type,
                key,
                iteration=trainer.epoch,
                value=trainer.metrics[key]
            ) 

    # Log F1 Score data to ClearML
    for tipo in ('B', 'M'):
        if f'metrics/precision({tipo})' in trainer.metrics.keys() and f'metrics/recall({tipo})' in trainer.metrics.keys():
            precision = trainer.metrics[f'metrics/precision({tipo})']
            recall = trainer.metrics[f'metrics/recall({tipo})']
            f1 = 2*(precision*recall)/(precision + recall)
            Logger.current_logger().report_scalar(
                'F1-Score',
                f'F1({tipo})',
                iteration=trainer.epoch,
                value=f1
            )

    # Log fitness values to ClearML (it is also logged in 'train' default graphic)
    Logger.current_logger().report_scalar(
        'Fitness',
        'Fitness function',
        iteration=trainer.epoch + 1,
        value=trainer.fitness
    )

def main():
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        output_uri=True # To upload the model and weights to ClearML.
    )

    # model_size = 'x'
    model_variant = f'yolov8{model_size}-seg'
    # model_variant = f'yolov8{model_size}'
    task.set_parameter('model_variant', model_variant)

    if pretrained: 
        model = YOLO(f'{model_variant}.pt')
    else:
        model = YOLO(f'{model_variant}.yaml')
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end) # Add callback to upload metrics data to ClearML

    train_args =  dict(
        data=dataset,
        epochs=epochs,
        patience=patience,
        batch=batch,
        project=yolo_proj,
        name=yolo_name,
        seed=seed
    )

    if optimizer is not None: 
        train_args['optimizer'] = optimizer
    if config is not None:
        train_args['cfg'] = config
        task.connect_configuration(train_args['cfg'], 'Config_file')
    if patience is not None:
        train_args['patience'] = patience
    if lr is not None: 
        train_args['lr0'] = lr

    task.connect(train_args)

    task.connect_configuration(train_args['data'], 'Dataset yaml')

    result = model.train(**train_args)

    # model.val(project=f'{yolo_proj}/{yolo_name}', name='val')

    task.close()


if __name__=='__main__':
    main()