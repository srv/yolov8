import os

path = './dataset/folds/valid/labels/'

n = 0
for txt_file in os.listdir(path):
    if txt_file.endswith('.txt'):
        empty = os.path.getsize(f'{path}/{txt_file}') == 0
        if empty: 
            n += 1
    
print(f'{n} empty files in {path}')