# -*- coding: utf-8 -*-

"""
tasks: ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels',]
py3 val.py sleeve_length_labels
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

validation_path = "/data/david/fai_attr/transfered_data/ROUND1/val_v6"
solver = Solver(validation_path=validation_path)

if len(sys.argv) == 2:
    task = sys.argv[1]
    assert task in task_list, "UNKOWN TASK"
    details = model_dict[task]

    utils.setup_log("%s-%s-%s-%s" % ('validating', task, details['network'], details['loss_type']))
    logging.info("start to validate single task: %s\n validation path: %s, parameters: %s" % (task, validation_path, details))

    val_acc, val_map, val_loss = solver.validate(None, model_path=details['model_path'], task=task, network=details['network'], batch_size=details['batch_size'], num_workers=details['num_workers'], gpus=details['gpus'], loss_type=details['loss_type'])
    logging.info('[%s]\n [model: %s]\n [nework: %s] Val-acc: %.3f, mAP: %.3f, loss: %.3f\n' % (task, details['model_path'], details['network'], val_acc, val_map, val_loss))

else:
    utils.setup_log("%s" % ('validating-all-tasks'))
    logging.info("start to validate task: %s\n validation_path: %s" % (task, validation_path))

    val_acc_list, val_map_list, val_loss_list = [], [], []
    for index, task in enumerate(model_dict):
        details = model_dict[task]
        logging.info("start training single task: %s\n validation path: %s, parameters: %s" % (task, validation_path, details))

        val_acc, val_map, val_loss = solver.validate(None, model_path=details['model_path'], task=task, network=details['network'], batch_size=details['batch_size'], num_workers=details['num_workers'], gpus=details['gpus'], loss_type=details['loss_type'])
        val_acc_list.append(val_acc)
        val_map_list.append(val_map)
        val_loss_list.append(val_loss)

        logging.info('[task: %s, model: %s, nework: %s]\n Val-acc: %.3f, mAP: %.3f, loss: %.3f\n' % (task, details['model_path'], details['network'], val_acc, val_map, val_loss))

    mean_val_acc = sum(val_acc_list)/len(val_acc_list)
    mean_val_map = sum(val_map_list)/len(val_map_list)
    mean_val_loss = sum(val_loss_list)/len(val_loss_list)
    logging.info("mean acc: %.4f, mean map: %.4f, mean loss %.4f\n" % (mean_val_acc, mean_val_map, mean_val_loss))
