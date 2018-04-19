# -*- coding: utf-8 -*-

import time
from pathlib import Path
from solver import Solver


model_dict = {
    'collar_design_labels' : {
        # 'model_path': '/data/david/models/fai_attrbutes/v1/collar_design_labels-2018-04-18-12-01-epoch-39.params',
        'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-46-89909/collar_design_labels-2018-04-19-20-18-epoch-22.params',
        'network': 'densenet201',
        'gpu': 2
    },

    'skirt_length_labels' : {
        # 'model_path': '/data/david/models/fai_attrbutes/v1/skirt_length_labels-2018-04-18-13-14-epoch-39.params',
        'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-47-46779/skirt_length_labels-2018-04-19-20-20-epoch-9.params',
        'network': 'densenet201',
        'gpu': 3
    },

    'lapel_design_labels' : {
        # 'model_path': '/data/david/models/fai_attrbutes/v1/lapel_design_labels-2018-04-18-16-59-epoch-39.params',
        'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-47-82970/lapel_design_labels-2018-04-19-19-54-epoch-19.params',
        'network': 'densenet201',
        'gpu': 6
    },

    'neckline_design_labels' : {
        # 'model_path': '/data/david/models/fai_attrbutes/v1/neckline_design_labels-2018-04-18-12-55-epoch-39.params',
        'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-48-6310/neckline_design_labels-2018-04-19-20-35-epoch-12.params',
        'network': 'densenet201',
        'gpu': 7
    },

    'coat_length_labels' : {
        # 'model_path': '/data/david/models/fai_attrbutes/v1/coat_length_labels-2018-04-18-16-29-epoch-39.params',
        'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-48-14548/coat_length_labels-2018-04-19-20-19-epoch-16.params',
        'network': 'densenet201',
        'gpu': 2
    },

    'neck_design_labels' : {
        # 'model_path': '/data/david/models/fai_attrbutes/v1/neck_design_labels-2018-04-18-14-57-epoch-39.params',
        'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-49-33957/neck_design_labels-2018-04-19-20-10-epoch-29.params',
        'network': 'densenet201',
        'gpu': 3
    },

    'pant_length_labels' : {
        # 'model_path': '/data/david/models/fai_attrbutes/v1/pant_length_labels-2018-04-18-17-59-epoch-38.params',
        'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-49-70385/pant_length_labels-2018-04-19-20-36-epoch-29.params',
        'network': 'densenet201',
        'gpu': 6
    },

    'sleeve_length_labels' : {
        # 'model_path': '/data/david/models/fai_attrbutes/v1/sleeve_length_labels-2018-04-18-14-50-epoch-39.params',
        'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-50-20482/sleeve_length_labels-2018-04-19-20-18-epoch-13.params',
        'network': 'densenet201',
        'gpu': 3
    }
}

test_dataset_path = '/data/david/fai_attr/transfered_data/test_v1'

# task = 'collar_design_labels'
# task = 'skirt_length_labels'
# task = 'lapel_design_labels'
# task = 'neckline_design_labels'
# task = 'coat_length_labels'
# task = 'neck_design_labels'
# task = 'pant_length_labels'
task = 'sleeve_length_labels'

details = model_dict[task]
batch_size=128
num_workers=16
solver = Solver(batch_size=batch_size, num_workers=num_workers, gpus=[details['gpu']], submission_path='/data/david/fai_attr/transfered_data/val_v1/')
solver.predict(test_dataset_path, model_path=details['model_path'], task=task, network=details['network'])

# solver = Solver(batch_size=128, num_workers=16, gpus=[7])
# for index, task in enumerate(model_dict):
#     details = model_dict[task]
#     solver.predict(test_dataset_path, model_path=details['model_path'], task=task, network=details['network'])
