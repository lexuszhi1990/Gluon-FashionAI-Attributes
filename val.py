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

VERSION = 'v2'
model_dict = config.MODEL_LIST[VERSION]
task_list = ['collar_design_labels', 'skirt_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'coat_length_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']

validation_path = '/data/david/fai_attr/transfered_data/val_v1'

batch_size=8
num_workers=4
gpus = [6]

results_file_path = Path('./results/results_roadmap.md')
f_out = results_file_path.open('a')
f_out.write('%s :\n' % time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())))
f_out.write('test path %s :\n' % validation_path)

solver = Solver(batch_size=batch_size, num_workers=num_workers, gpus=gpus, validation_path=validation_path)

if len(sys.argv) == 2:
    task = sys.argv[1]
    print("start predict single task: %s" % task)
    assert task in task_list, "UNKOWN TASK"
    details = model_dict[task]

    val_acc, val_map, val_loss = solver.validate(None, model_path=details['model_path'], task=task, network=details['network'])
    print('[%s]\n [model: %s]\n [nework: %s] Val-acc: %.3f, mAP: %.3f, loss: %.3f\n' % (task, details['model_path'], details['network'], val_acc, val_map, val_loss))
    f_out.write('[%s]\n [model: %s]\n [nework: %s] Val-acc: %.3f, mAP: %.3f, loss: %.3f\n' % (task, details['model_path'], details['network'], val_acc, val_map, val_loss))
else:
    val_acc_list, val_map_list, val_loss_list = [], [], []
    for index, task in enumerate(model_dict):
        details = model_dict[task]
        val_acc, val_map, val_loss = solver.validate(None, model_path=details['model_path'], task=task, network=details['network'])
        val_acc_list.append(val_acc)
        val_map_list.append(val_map)
        val_loss_list.append(val_loss)

        f_out.write('[%s]\n [model: %s]\n [nework: %s] Val-acc: %.3f, mAP: %.3f, loss: %.3f\n' % (task, details['model_path'], details['network'], val_acc, val_map, val_loss))

    mean_val_acc = sum(val_acc_list)/len(val_acc_list)
    mean_val_map = sum(val_map_list)/len(val_map_list)
    mean_val_loss = sum(val_loss_list)/len(val_loss_list)
    print("mean acc: %.4f, mean map: %.4f, mean loss %.4f" % (mean_val_acc, mean_val_map, mean_val_loss))
    f_out.write("mean acc: %.4f, mean map: %.4f, mean loss %.4f\n" % (mean_val_acc, mean_val_map, mean_val_loss))

f_out.close()
