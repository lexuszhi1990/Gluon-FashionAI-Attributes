# -*- coding: utf-8 -*-
# usage:
# task = 'collar_design_labels'
# task = 'skirt_length_labels'
# task = 'lapel_design_labels'
# task = 'neckline_design_labels'
# task = 'coat_length_labels'
# task = 'neck_design_labels'
# task = 'pant_length_labels'
# task = 'sleeve_length_labels'
# py3 train.py neck_design_labels

import sys, os
import time
from pathlib import Path
from solver import Solver
import numpy as np
import logging

from src import utils
from src.config import config

training_path = "/data/david/fai_attr/transfered_data/train_v6"
validation_path = "/data/david/fai_attr/transfered_data/val_v6"
ckpt_path = '/data/david/models/fai_attrbutes/v2'

VERSION = 'v4'
model_dict = config.MODEL_LIST[VERSION]
task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']

# os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = 0
# os.environ['CUDA_VISIBLE_DEVICES'] = str(details['gpu'])

solver = Solver(training_path=training_path, validation_path=validation_path, ckpt_path=ckpt_path)
if len(sys.argv) == 2:
    task = sys.argv[1]
    assert task in task_list, "UNKOWN TASK"
    details = model_dict[task]

    utils.setup_log("%s-%s-%s-%s" % ('training', task, details['network'], details['loss_type']))
    logging.info("start training task: %s\n parameters: %s\n training_path: %s, validation_path: %s" % (task, details, training_path, validation_path))

    solver.train(task=task, network=details['network'], epochs=details['epochs'], lr=details['lr'], momentum=details['momentum'], wd=details['wd'], lr_factor=details['lr_factor'], lr_steps=details['lr_steps'], gpus=details['gpus'], batch_size=details['batch_size'], num_workers=details['num_workers'], loss_type=details['loss_type'])
