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

VERSION = 'v3'
model_dict = config.MODEL_LIST[VERSION]
task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']

task = sys.argv[1]
assert task in task_list, "UNKOWN TASK"

batch_size = 4
num_workers = max(int(batch_size/2), 1)
lr = 0.001
lr_factor = 0.75
epochs = 20
lr_steps = [5,10,15,np.inf]
wd = 1e-4

momentum = 0.9

details = model_dict[task]
os.environ['CUDA_VISIBLE_DEVICES'] = str(details['gpu'])

solver = Solver(batch_size=details['batch_size'], num_workers=details['num_workers'], gpus=[0])
solver.train(task=task, model_name=details['network'], epochs=epochs, lr=details['lr'], momentum=momentum, wd=wd, lr_factor=lr_factor, lr_steps=lr_steps)
