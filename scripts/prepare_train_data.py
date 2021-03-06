import mxnet
from mxnet import gluon, image
import os, shutil, random

# Read label.csv
# For each task, make folders, and copy picture to corresponding folders

root_path = '/data/fashion/data/attribute/datasets_david'
label_dir = os.path.join(root_path, 'base', 'Annotations/label.csv')
warmup_label_dir = os.path.join(root_path, 'web', 'Annotations/skirt_length_labels.csv')

data_path = '/data/david/fai_attr/gloun_data/train_valid'
submission_path = '/data/david/fai_attr/gloun_data/submission'

label_dict = {'coat_length_labels': [],
              'lapel_design_labels': [],
              'neckline_design_labels': [],
              'skirt_length_labels': [],
              'collar_design_labels': [],
              'neck_design_labels': [],
              'pant_length_labels': [],
              'sleeve_length_labels': []}

task_list = label_dict.keys()

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

mkdir_if_not_exist([data_path])

with open(label_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        label_dict[task].append((path, label))

for task, path_label in label_dict.items():
    mkdir_if_not_exist([data_path,  task])
    train_count = 0
    n = len(path_label)
    m = len(list(path_label[0][1]))

    for mm in range(m):
        mkdir_if_not_exist([data_path, task, 'train', str(mm)])
        mkdir_if_not_exist([data_path, task, 'val', str(mm)])

    random.shuffle(path_label)
    for path, label in path_label:
        label_index = list(label).index('y')
        src_path = os.path.join(root_path, 'base', path)
        if train_count < n * 0.9:
            dest_path = os.path.join(data_path, task, 'train', str(label_index))
        else:
            dest_path = os.path.join(data_path, task, 'val', str(label_index))
        shutil.copy(src_path, dest_path)
        print("copy %s to %s" % (src_path, dest_path))
        train_count += 1

# Add warmup data to skirt task

label_dict = {'skirt_length_labels': []}

with open(warmup_label_dir, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        label_dict[task].append((path, label))

for task, path_label in label_dict.items():
    train_count = 0
    n = len(path_label)
    m = len(list(path_label[0][1]))

    random.shuffle(path_label)
    for path, label in path_label:
        label_index = list(label).index('y')
        src_path = os.path.join(root_path, 'web', path)
        if train_count < n * 0.9:
            dest_path = os.path.join(data_path, task, 'train', str(label_index))
        else:
            dest_path = os.path.join(data_path, task, 'val', str(label_index))
        shutil.copy(src_path, dest_path)
        print("copy %s to %s" % (src_path, dest_path))
        train_count += 1
