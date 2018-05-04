# -*- coding: utf-8 -*-

"""
tasks: ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels',]
py3 train.py skirt_length_labels
"""

import sys, os
import time
from pathlib import Path
from solver import Solver
import numpy as np
import logging

from src import utils
from src.config import config

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(details['gpu'])

training_path = "/data/david/fai_attr/transfered_data/ROUND2/PURE_TRAIN_V1.2"
validation_path = "/data/david/fai_attr/transfered_data/ROUND1/val_v7"
# training_path = "/data/david/fai_attr/transfered_data/ROUND1/train_v6"
# validation_path = "/data/david/fai_attr/transfered_data/ROUND1/val_v6"

# ckpt_path = '/data/david/fai_attr/submissions/round2/v0.1'
ckpt_path = None

VERSION = 'v4'
model_dict = config.MODEL_LIST[VERSION]
task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']

solver = Solver(training_path=training_path, validation_path=validation_path, ckpt_path=ckpt_path)
if len(sys.argv) == 2:
    task = sys.argv[1]
    assert task in task_list, "UNKOWN TASK"
    details = model_dict[task]
    batch_size = details['batch_size'] * max(len(details['gpus']), 1)

    utils.setup_log("%s-%s-%s-%s" % ('training', task, details['network'], details['loss_type']))
    logging.info("start training task: %s\n parameters: %s\n training_path: %s, validation_path: %s" % (task, details, training_path, validation_path))

    solver.train(task=task, network=details['network'], epochs=details['epochs'], lr=details['lr'], momentum=details['momentum'], wd=details['wd'], lr_factor=details['lr_factor'], lr_steps=details['lr_steps'], gpus=details['gpus'], batch_size=batch_size, num_workers=details['num_workers'], loss_type=details['loss_type'], model_path=details['model_path'], resume=True)
