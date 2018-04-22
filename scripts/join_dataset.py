

### combine dataset

from pathlib import Path
import json

train_coco_format_file = 'train_warmup.json'
train_detection_file = 'train_warmup_results-max-all.json'
val_coco_format_file = 'validation_v1.json'
val_detection_file = 'validation_v1_detection_max_5.json'
train_val_coco_format_file = 'train_validation_v1.json'
train_val_detection_file = 'train_validation_v1_detection_max_5.json'
train_list = json.load(Path(train_coco_format_file).open('r'))
val_list = json.load(Path(val_coco_format_file).open('r'))
train_dets = json.load(Path(train_detection_file).open('r'))
val_dets = json.load(Path(val_detection_file).open('r'))

train_list['images'].extend(val_list['images'])
json.dump(train_list, Path(train_val_coco_format_file).open('w+'))
train_dets.extend(val_dets)
json.dump(train_dets, Path(train_val_detection_file).open('w+'))

train_val_dets = json.load(Path(train_val_detection_file).open('r'))

# train_val_list = {}
# train_val_dets = {}

# # 'images', 'categories', 'annotations'
# # train_list['categories'].extend(val_list['categories'])
# # train_list['annotations'].extend(val_list['annotations'])

# train_val_list['images'] = train_list['images']
# train_val_list['categories'] = train_list['categories']
# train_val_list['annotations'] = train_list['annotations']

# check results
from collections import Counter

label_file_op = Path(label_file_path).open('r')
lines = label_file_op.readlines()
tokens = [l.rstrip().split(',') for l in lines]
