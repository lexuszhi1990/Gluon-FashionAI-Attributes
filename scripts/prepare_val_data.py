import mxnet
from mxnet import gluon, image
import os, shutil, random
from pathlib import Path

# Read label.csv
# For each task, make folders, and copy picture to corresponding folders

source_dataset_path = '/data/fashion/data/attribute/datasets_david'
dst_data_path = '/data/david/fai_attr/gloun_data/train_valid'
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

mkdir_if_not_exist([dst_data_path])
mkdir_if_not_exist([submission_path])

for task in task_list:
    label_path = Path(source_dataset_path, 'base', 'Annotations/%s_val.txt'%task)
    if not label_path.exists():
        print("%s not exists" % label_path)
        continue

    lines = label_path.open('r').readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        label_dict[task].append((path, label))
    print("add task %s" % task)

for task, path_label in label_dict.items():
    mkdir_if_not_exist([dst_data_path, task])
    train_count = 0
    n = len(path_label)
    m = len(list(path_label[0][1]))

    for mm in range(m):
        mkdir_if_not_exist([dst_data_path, task, 'train', str(mm)])
        mkdir_if_not_exist([dst_data_path, task, 'val', str(mm)])

    for path, label in path_label:
        label_index = list(label).index('y')
        src_path = os.path.join(source_dataset_path, 'rank', path)
        dst_path = os.path.join(dst_data_path, task, 'val', str(label_index))
        shutil.copy(src_path, dst_path)
        print('copy img from %s to %s' % (src_path, dst_path))
