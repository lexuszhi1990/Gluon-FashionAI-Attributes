# -*- coding: utf-8 -*-

"""
tasks: ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels',]
py3 predict.py skirt_length_labels
"""

import time
import sys
import logging
from pathlib import Path
from solver import Solver
from src import utils
from src.config import config

VERSION = 'v4'
model_dict = config.MODEL_LIST[VERSION]
task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']

test_dataset_path = "/data/david/fai_attr/transfered_data/ROUND2/RANK_V1.1"
submission_path = '/data/david/fai_attr/submissions/round2/v1.2'

# gpus = None
gpus = [2]
cropped_predict=False

solver = Solver(submission_path=submission_path)
if len(sys.argv) == 2:
    task = sys.argv[1]
    assert task in task_list, "UNKOWN TASK"
    details = model_dict[task]

    utils.setup_log("%s-%s-%s-%s" % ('predicting', task, details['network'], details['loss_type']))
    logging.info("start to predict task: %s\n test_dataset_path: %s, parameters: %s" % (task, test_dataset_path, details))
    current_gpus = details['gpus'] if gpus is None else gpus
    solver.predict(test_dataset_path, model_path=details['model_path'], task=task, gpus=current_gpus, network=details['network'], cropped_predict=cropped_predict, loss_type=details['loss_type'])
else:
    utils.setup_log("%s" % ('predict-all-tasks'))
    for index, task in enumerate(model_dict):
        details = model_dict[task]
        logging.info("start to predict task: %s\n test_dataset_path: %s, parameters: %s" % (task, test_dataset_path, details))
        current_gpus = details['gpus'] if gpus is None else gpus
        solver.predict(test_dataset_path, model_path=details['model_path'], task=task, gpus=current_gpus, network=details['network'], cropped_predict=cropped_predict, loss_type=details['loss_type'])
