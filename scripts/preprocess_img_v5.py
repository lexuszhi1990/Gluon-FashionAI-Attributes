import sys
sys.path.append('/home/david/fashionAI/fai_attr/Gluon-FashionAI-Attributes/data/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import json
import csv
import cv2
import imutils
from pathlib import Path
import pickle
import numpy as np

BASE_SHAPE = 360

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

label_dict = {'coat_length_labels': [],
              'lapel_design_labels': [],
              'neckline_design_labels': [],
              'skirt_length_labels': [],
              'collar_design_labels': [],
              'neck_design_labels': [],
              'pant_length_labels': [],
              'sleeve_length_labels': []}

transfered_label_dict = {'coat_length_labels': [],
              'lapel_design_labels': [],
              'neckline_design_labels': [],
              'skirt_length_labels': [],
              'collar_design_labels': [],
              'neck_design_labels': [],
              'pant_length_labels': [],
              'sleeve_length_labels': []}

# Train
dataset_path = '/data/david/fai_attr/raw_data/ROUND2/PURE_TRAIN_V1'
dataset_json_file = dataset_path + '/Annotations/label_coco.json'
results_json_file = dataset_path + '/Annotations/detections_label_coco_results.json'
label_file_path = dataset_path + '/Annotations/label_without_head.csv'

# with new crop methods
# outout_path = '/data/david/fai_attr/transfered_data/ROUND2/PURE_TRAIN_V1.2'
# 1.3: remove the not exists samples
outout_path = '/data/david/fai_attr/transfered_data/ROUND2/PURE_TRAIN_V1.3'

# dataset_json_file = '/data/david/fai_attr/bak/gloun_data/detection_labels/validation_v1.json'
# results_json_file = '/data/david/fai_attr/bak/gloun_data/detection_labels/validation_v1_detection_max_5.json'
# dataset_path = '/data/david/fai_attr/raw_data/ROUND1/val_v1'
# label_file_path = dataset_path + '/Annotations/val.csv'
# outout_path = '/data/david/fai_attr/transfered_data/ROUND1/val_v7'

# dataset_path = '/data/david/fai_attr/raw_data/ROUND2/RANK_V1'
# dataset_json_file = dataset_path + '/Tests/question_coco.json'
# results_json_file = dataset_path + '/Tests/detections_question_coco_results.json'
# label_file_path = dataset_path + '/Tests/question_origin.csv'
# outout_path = '/data/david/fai_attr/transfered_data/ROUND2/RANK_V1.1'

for file_path in [dataset_json_file, results_json_file, label_file_path]:
    assert Path(file_path).exists(), "%s not exist" % file_path

def find_label_by_path(task, img_path):
    for line in label_dict[task]:
        if img_path == line[0]:
            return line[1]
    return None

def convert_label_to_one_hot(raw_label):
    label_y = [int(l == 'y')*1 for l in raw_label]
    label_m = [int(l == 'm')*0.5 for l in raw_label]
    label = [x+y for x,y in zip(label_m, label_y)]
    return label

def normalize_image(data):
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std

coco=COCO(dataset_json_file)
detections = json.load(Path(results_json_file).open())
# [x['name'] for x in coco.cats.values()]
cat_list = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
task_list = ['coat_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'skirt_length_labels', 'collar_design_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']
total_nums = 0
no_dets_nums = 0
match_nums = 0
not_exist_nums = 0

label_file_op = Path(label_file_path).open('r')
lines = label_file_op.readlines()
tokens = [l.rstrip().split(',') for l in lines]

for img_relative_path, task, label in tokens:

    if task != 'sleeve_length_labels':
        continue

    label_dict[task].append((img_relative_path, label))

    img_infos = [img_info for img_info in coco.imgs.values() if img_info['file_name'] == img_relative_path]
    assert len(img_infos) >= 1
    img_info = img_infos[0]
    img_path = Path(dataset_path, img_info['file_name'])

    if not img_path.exists():
        continue

    assert img_path.exists(), "img_path %s not exists" % img_path
    img_raw = cv2.imread(img_path.as_posix())
    img_raw_height, img_raw_width = img_raw.shape[:2]
    raw_label = find_label_by_path(task, img_info['file_name'])

    if raw_label is None:
        continue
    assert raw_label is not None, "raw_label is None"

    if raw_label[0] == 'y':
        not_exist_nums += 1
        continue

    one_hot_label = convert_label_to_one_hot(raw_label)
    dets = [det for det in detections if det['image_id'] == img_info['id']]
    if len(dets) == 0:
        img_raw_resieze = cv2.resize(img_raw, (BASE_SHAPE, BASE_SHAPE))
        cv2.imwrite(Path(outout_path, img_info['file_name']).as_posix(), img_raw_resieze)
        transfered_label_dict[task].append((img_info['file_name'], task, raw_label, one_hot_label, [0, 0, 0, 0], "UNKNOWN"))
        no_dets_nums = no_dets_nums+1
        print("no dets for %s " % img_info['file_name'])
        continue

    # choose best det here
    # task_list = ['coat_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'skirt_length_labels', 'collar_design_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']
    det_names = [coco.cats[det['category_id']]['name'] for det in dets]
    if task in ['sleeve_length_labels', 'coat_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'collar_design_labels', 'neck_design_labels']:
        for cat_name in ["outwear", 'blouse', 'dress']:
            if cat_name in det_names:
                curent_det = dets[det_names.index(cat_name)]
    elif task in ['skirt_length_labels', 'pant_length_labels']:
        for cat_name in ["skirt", 'trousers', 'dress']:
            if cat_name in det_names:
                curent_det = dets[det_names.index(cat_name)]

    if curent_det is None:
        det = dets[0]
        print('category %s not match task: %s' % (coco.cats[det['category_id']]['name'], task))
    else:
        det = curent_det
        match_nums = match_nums + 1

    category_id = det['category_id']
    category_name = coco.cats[det['category_id']]['name']

    # refine bbox according by task
    bbox = [int(i) for i in det['bbox']]
    # expand the size to 30% according by the width

    if task == "neck_design_labels":
        bbox[0] = max(bbox[0] - bbox[2] * 0.3, 0)
        bbox[2] = min(img_raw_width, bbox[2] * 1.5)
        if bbox[1] > img_raw_height * 0.2:
            bbox[1] = img_raw_height * 0.1
        else:
            bbox[1] = 0
        bbox[3] = min(img_raw_height-bbox[1]-1, bbox[3] * 1.2)
    elif task == "neckline_design_labels":
        bbox[0] = max(bbox[0] - bbox[2] * 0.3, 0)
        bbox[2] = min(img_raw_width, bbox[2] * 1.5)
        if bbox[1] > img_raw_height * 0.2:
            bbox[1] = img_raw_height * 0.1
        else:
            bbox[1] = 0
        bbox[3] = min(img_raw_height-bbox[1]-1, bbox[3] * 1.2)
    elif task == "collar_design_labels":
        bbox[0] = max(bbox[0] - bbox[2] * 0.3, 0)
        bbox[2] = min(img_raw_width, bbox[2] * 1.5)
        if bbox[1] > img_raw_height * 0.2:
            bbox[1] = img_raw_height * 0.1
        else:
            bbox[1] = 0
        bbox[3] = min(img_raw_height-bbox[1]-1, bbox[3] * 1.3)
    elif task == "lapel_design_labels":
        bbox[0] = max(bbox[0] - bbox[2] * 0.3, 0)
        bbox[2] = min(img_raw_width, bbox[2] * 1.5)
        if bbox[1] > img_raw_height * 0.3:
            bbox[1] = img_raw_height * 0.1
        else:
            bbox[1] = 0
        bbox[3] = min(img_raw_height-bbox[1]-1, bbox[3] * 1.4)

    elif task in ['sleeve_length_labels']:
        # 6e4360353852141ecf8d51f3d2706be3.jpg
        bbox[0] = max(bbox[0] - bbox[2] * 0.3, 0)
        bbox[2] = min(img_raw_width, bbox[2] * 1.5)

        if bbox[1] > img_raw_height * 0.3:
            bbox[1] = img_raw_height * 0.1
        else:
            bbox[1] = max(0, bbox[1] - bbox[3] * 0.5)

        bbox[3] = min(img_raw_height-bbox[1]-1, bbox[3] * 1.5)

    elif task in ['coat_length_labels']:
        bbox[0] = max(bbox[0] - bbox[2] * 0.3, 0)
        bbox[2] = min(img_raw_width, bbox[2] * 1.5)

        if bbox[1] > img_raw_height * 0.3:
            bbox[1] = max(0, img_raw_height * 0.1)
        else:
            bbox[1] = max(0, bbox[1] - bbox[3] * 0.4)

        bbox[3] = min(img_raw_height-bbox[1]-1, bbox[3] * 1.5)

    elif task in ['skirt_length_labels']:
        # wrong image id: ['69fc8936d8c04f2e30796ea90138966a.jpg']
        bbox[0] = max(bbox[0] - bbox[2] * 0.3, 0)
        bbox[2] = min(img_raw_width, bbox[2] * 1.5)

        if bbox[1] > img_raw_height * 0.4:
            bbox[1] = img_raw_height * 0.2
        else:
            bbox[1] = max(0, bbox[1] - bbox[3] * 0.3)

        if bbox[3]+bbox[1] <= img_raw_height * 0.8:
            bbox[3] = (img_raw_height-bbox[1]-1)*0.90
        else:
            bbox[3] = (img_raw_height-bbox[1]-1)*0.99

    elif task in ['pant_length_labels']:
        bbox[0] = max(bbox[0] - bbox[2] * 0.3, 0)
        bbox[2] = min(img_raw_width, bbox[2] * 1.5)

        if bbox[1] > img_raw_height * 0.4:
            bbox[1] = img_raw_height * 0.2
        else:
            bbox[1] = max(0, bbox[1] - bbox[3] * 0.3)
        bbox[3] = (img_raw_height-bbox[1]-1)*0.99
    else:
        raise RuntimeError

    bbox = [int(i) for i in bbox]

    cropped_img = img_raw[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    c_h, c_w = cropped_img.shape[:2]
    max_dim = max(c_h, c_w)
    top_pad = (max_dim - c_h) // 2
    bottom_pad = max_dim - c_h - top_pad
    left_pad = (max_dim - c_w) // 2
    right_pad = max_dim - c_w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    cropped_img = np.pad(cropped_img, padding, mode='constant', constant_values=0)
    resize_img = cv2.resize(cropped_img, (BASE_SHAPE, BASE_SHAPE), interpolation=cv2.INTER_CUBIC)

    # post processing
    output_imgs_path = Path(outout_path, 'Images', task)
    if not output_imgs_path.exists():
        output_imgs_path.mkdir(parents=True)
    # 1. save image
    output_resized_img_path = Path(output_imgs_path, img_path.name)
    cv2.imwrite(output_resized_img_path.as_posix(), resize_img)
    print("save image at %s" % output_resized_img_path)
    # print("save image to %s as shape (%d, %d)" % (output_resized_img_path, resize_img.shape[0], resize_img.shape[1]))
    # 2. save new labels
    transfered_label_dict[task].append((img_info['file_name'], task, raw_label, one_hot_label, bbox, category_name))
    # 3. TODO: save new labels to pkl
    # output_label_pkl_path = Path(output_imgs_path, img_path.stem+'.pkl')
    # total_results = {"image": resize_img.astype(float), "raw_label": raw_label, "one_hot_label": one_hot_label}
    # pickle.dump(total_results, output_label_pkl_path.open("w"))
    total_nums += 1

print("total matched %d/%d" % (match_nums, total_nums))
print("no dets matched %d/%d" % (no_dets_nums, total_nums))
print("no exits number %d/%d" % (not_exist_nums, total_nums))

for task in transfered_label_dict.keys():
    csv_file_path = Path(outout_path, 'Annotations', "%s.csv" % task)
    if not csv_file_path.parent.exists():
        csv_file_path.parent.mkdir(parents=True)

    csv_file = csv_file_path.open('w+')
    spamwriter = csv.writer(csv_file)

    for line in transfered_label_dict[task]:
        file_name, task, raw_label, one_hot_label, bbox, category_name = line
        output_label_list = '_'.join([str(x) for x in one_hot_label])
        output_bbox_list = '_'.join([str(x) for x in bbox])
        spamwriter.writerow([file_name, task, raw_label, output_label_list, output_bbox_list, category_name])

    csv_file.close()
    print("finished writing %s " % csv_file_path)

# coco.imgs:
# 4073319441911775229: {'id': 4073319441911775229, 'height': 512, 'width': 512, 'file_name': 'Images/skirt_length_labels/14cff0d4c1566b53937556a56a396cb0.jpg'}
# detection
# det = detections[0]
# {'bbox': [141.21919476310723, 148.1283797967682, 268.367932953227, 336.9712621340266], 'score': 0.16471093893051147, 'image_id': 7322388807026071477, 'category_id': 1}
# (Pdb) len(coco.imgs)
#   89683
# [('blouse', 19387), ('dress', 31065), ('outwear', 18212), ('skirt', 13191), ('trousers', 7822)]

