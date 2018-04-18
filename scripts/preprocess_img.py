import sys
sys.path.append('/home/david/fashionAI/Gluon-FashionAI-Attributes/data/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import json
from pathlib import Path


import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import pylab
import json
import csv
import cv2

from pathlib import Path

dataset_json_file = '/data/david/fai_attr/gloun_data/detection_labels/train_val.json'
results_json_file = '/data/david/fai_attr/gloun_data/detection_labels/train_val_results-v1.json'

coco=COCO(dataset_json_file)
detections = json.load(Path(results_json_file).open())

det = detections[0]
cat_name = coco.cats[det['category_id']]['name']

img_anno = coco.loadImgs(det['image_id'])[0]
img_path = Path(output_dir, 'images', image_set, img_anno['file_name'])
