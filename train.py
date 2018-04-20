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
from src.config import config

VERSION = 'v4'
model_dict = config.MODEL_LIST[VERSION]
task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']
training_path = "/data/david/fai_attr/transfered_data/train_v4"
# os.environ['CUDA_VISIBLE_DEVICES'] = str(details['gpu'])

if len(sys.argv) == 2:
    task = sys.argv[1]
    assert task in task_list, "UNKOWN TASK"
    print("start predict single task: %s" % task)
    details = model_dict[task]
    solver = Solver(training_path=training_path)
    solver.train(task=task, model_name=details['network'], epochs=details['epochs'], lr=details['lr'], momentum=details['momentum'], wd=details['wd'], lr_factor=details['lr_factor'], lr_steps=details['lr_steps'], gpus=details['gpus'], batch_size=details['batch_size'], num_workers=details['num_workers'])
