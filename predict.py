# -*- coding: utf-8 -*-

import time
from pathlib import Path
from solver import Solver


model_dict = {
    'collar_design_labels' : {
        'model_path': '/data/david/models/fai_attrbutes/v1/collar_design_labels-2018-04-18-12-01-epoch-39.params',
        'network': 'densenet201'
    },

    'skirt_length_labels' : {
        'model_path': '/data/david/models/fai_attrbutes/v1/skirt_length_labels-2018-04-18-13-14-epoch-39.params',
        'network': 'densenet201'
    },

    'lapel_design_labels' : {
        'model_path': '/data/david/models/fai_attrbutes/v1/lapel_design_labels-2018-04-18-16-59-epoch-39.params',
        'network': 'densenet201'
    },

    'neckline_design_labels' : {
        'model_path': '/data/david/models/fai_attrbutes/v1/neckline_design_labels-2018-04-18-12-55-epoch-39.params',
        'network': 'densenet201'
    },

    'coat_length_labels' : {
        'model_path': '/data/david/models/fai_attrbutes/v1/coat_length_labels-2018-04-18-16-29-epoch-39.params',
        'network': 'densenet201'
    },

    'neck_design_labels' : {
        'model_path': '/data/david/models/fai_attrbutes/v1/neck_design_labels-2018-04-18-14-57-epoch-39.params',
        'network': 'densenet201'
    },

    'pant_length_labels' : {
        'model_path': '/data/david/models/fai_attrbutes/v1/pant_length_labels-2018-04-18-17-59-epoch-38.params',
        'network': 'densenet201'
    },

    'sleeve_length_labels' : {
        'model_path': '/data/david/models/fai_attrbutes/v1/sleeve_length_labels-2018-04-18-14-50-epoch-39.params',
        'network': 'densenet201'
    }
}

test_dataset_path = '/data/david/fai_attr/gloun_data/test/z_rank'
solver = Solver(batch_size=24, num_workers=16, gpus=[7], solver_type='Validate')
for index, task in enumerate(model_dict):
    details = model_dict[task]
    # def predict(self, dataset_path, model_path, task, network='densenet201'):
    solver.predict(test_dataset_path, model_path=details['model_path'], task=task, network=details['network'])
