# -*- coding: utf-8 -*-
# task = 'collar_design_labels'
# task = 'skirt_length_labels'
# task = 'lapel_design_labels'
# task = 'neckline_design_labels'
# task = 'coat_length_labels'
# task = 'neck_design_labels'
# task = 'pant_length_labels'
# task = 'sleeve_length_labels'

import time
import sys
from pathlib import Path
from solver import Solver
from src.config import config

VERSION = 'v1'
model_dict = config.MODEL_LIST[VERSION]
task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']

test_dataset_path = '../data/z_rank'
submission_path = '../submit/v1'

gpus = [2]
cropped_predict=True

solver = Solver(submission_path=submission_path)
if len(sys.argv) == 2:
    task = sys.argv[1]
    print("start predict single task: %s" % task)
    assert task in task_list, "UNKOWN TASK"
    details = model_dict[task]
    # predict(dataset_path, model_path, task, gpus, network='densenet201', cropped_predict=True)
    solver.predict(test_dataset_path, model_path=details['model_path'], task=task, gpus=gpus, network=details['network'], cropped_predict=cropped_predict)
else:
    for index, task in enumerate(model_dict):
        details = model_dict[task]
        solver.predict(test_dataset_path, model_path=details['model_path'], task=task, gpus=gpus, network=details['network'], cropped_predict=cropped_predict)
