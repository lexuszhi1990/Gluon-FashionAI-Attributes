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

VERSION = 'v3'
model_dict = config.MODEL_LIST[VERSION]
task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']

test_dataset_path = '/data/david/fai_attr/transfered_data/test_v1'
# submission_path = '/data/david/fai_attr/transfered_data/test_v1/submissions/test_v2'
submission_path = '/data/david/fai_attr/gloun_data/submission/2018-04-20-11-42-74877'

batch_size=4
num_workers=2
gpus = [0]

if len(sys.argv) == 2:
    task = sys.argv[1]
    print("start predict single task: %s" % task)
    assert task in task_list, "UNKOWN TASK"
    details = model_dict[task]
    solver = Solver(batch_size=batch_size, num_workers=num_workers, gpus=gpus, submission_path=submission_path)
    solver.predict(test_dataset_path, model_path=details['model_path'], task=task, network=details['network'])
else:
    solver = Solver(batch_size=batch_size, num_workers=16, gpus=gpus, submission_path=submission_path)
    for index, task in enumerate(model_dict):
        details = model_dict[task]
        solver.predict(test_dataset_path, model_path=details['model_path'], task=task, network=details['network'])
