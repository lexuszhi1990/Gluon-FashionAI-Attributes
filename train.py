# -*- coding: utf-8 -*-

import time
from pathlib import Path
from solver import Solver
import numpy as np

model_dict = {
    'collar_design_labels' : {
        'network': 'densenet201',
        'gpu': 0
    },

    'skirt_length_labels' : {
        'network': 'densenet201',
        'gpu': 1
    },

    'lapel_design_labels' : {
        'network': 'densenet201',
        'gpu': 2
    },

    'neckline_design_labels' : {
        'network': 'densenet201',
        'gpu': 3
    },

    'coat_length_labels' : {
        'network': 'densenet201',
        'gpu': 4
    },

    'neck_design_labels' : {
        'network': 'densenet201',
        'gpu': 5
    },

    'pant_length_labels' : {
        'network': 'densenet201',
        'gpu': 6
    },

    'sleeve_length_labels' : {
        'network': 'densenet201',
        'gpu': 7
    }
}

# task_list = ['collar_design_labels']

# task = 'collar_design_labels'
# task = 'skirt_length_labels'
# task = 'lapel_design_labels'
# task = 'neckline_design_labels'
# task = 'coat_length_labels'
# task = 'neck_design_labels'
# task = 'pant_length_labels'
task = 'sleeve_length_labels'

epochs = 30
num_workers = 14
batch_size = 12
lr = 0.001
lr_factor = 0.75
lr_steps = [5,10,20,np.inf]
wd = 1e-4
momentum = 0.9

details = model_dict[task]
solver = Solver(batch_size=batch_size, num_workers=num_workers, gpus=[details['gpu']])
solver.train(task=task, model_name=details['network'], epochs=epochs, lr=lr, momentum=momentum, wd=wd, lr_factor=lr_factor, lr_steps=lr_steps)
