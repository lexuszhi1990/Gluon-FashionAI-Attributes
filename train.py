# -*- coding: utf-8 -*-

import time
from pathlib import Path
from solver import Solver

model_dict = {
    'collar_design_labels' : {
        'network': 'densenet201',
        'gpu': 4
    },

    'skirt_length_labels' : {
        'network': 'densenet201',
        'gpu': 5
    },

    'lapel_design_labels' : {
        'network': 'densenet201',
        'gpu': 6
    },

    'neckline_design_labels' : {
        'network': 'densenet201',
        'gpu': 7
    },

    'coat_length_labels' : {
        'network': 'densenet201',
        'gpu': 4
    },

    'neck_design_labels' : {
        'network': 'densenet201',
        'gpu': 4
    },

    'pant_length_labels' : {
        'network': 'densenet201',
        'gpu': 4
    },

    'sleeve_length_labels' : {
        'network': 'densenet201',
        'gpu': 4
    }
}

task_list = ['collar_design_labels']

epochs = 40
num_workers = 16
batch_size = 24
lr = 0.001
lr_factor = 0.75
lr_steps = [10,20,30]
wd = 1e-4
momentum = 0.9

for task in task_list:
    details = model_dict[task]
    solver = Solver(batch_size=batch_size, num_workers=num_workers, gpus=[details['gpu']])
    solver.train(task=task, model_name=details['network'], epochs=epochs, lr=lr, momentum=momentum, wd=wd, lr_factor=lr_factor, lr_steps=lr_steps)
