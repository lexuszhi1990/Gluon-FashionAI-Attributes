# -*- coding: utf-8 -*-

from easydict import EasyDict as edict


config = edict()
config.MODEL_LIST = edict()

config.MODEL_LIST.V1 = edict()

config.MODEL_LIST = {
    'v1' : {
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
    },

    "v2": {
        'collar_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-46-89909/collar_design_labels-2018-04-19-20-18-epoch-22.params',
            'network': 'densenet201'
        },
        'skirt_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-47-46779/skirt_length_labels-2018-04-19-20-20-epoch-9.params',
            'network': 'densenet201'
        },
        'lapel_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-47-82970/lapel_design_labels-2018-04-19-19-54-epoch-19.params',
            'network': 'densenet201'
        },
        'neckline_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-48-6310/neckline_design_labels-2018-04-19-20-35-epoch-12.params',
            'network': 'densenet201'
        },
        'coat_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-48-14548/coat_length_labels-2018-04-19-20-19-epoch-16.params',
            'network': 'densenet201'
        },
        'neck_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-49-33957/neck_design_labels-2018-04-19-20-10-epoch-29.params',
            'network': 'densenet201'
        },
        'pant_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-49-70385/pant_length_labels-2018-04-19-20-36-epoch-29.params',
            'network': 'densenet201'
        },
        'sleeve_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-19-18-50-20482/sleeve_length_labels-2018-04-19-20-18-epoch-13.params',
            'network': 'densenet201'
        }
    },

    "v3": {
        'collar_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-10-32-34904/collar_design_labels-2018-04-20-11-26-epoch-19.params',
            'network': 'densenet121',
            'gpu' : 0,
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001

        },
        'skirt_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-10-58-52767/skirt_length_labels-2018-04-20-11-27-epoch-2.params',
            'network': 'densenet201',
            'gpu' : 1,
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001

        },
        'lapel_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-10-35-95574/lapel_design_labels-2018-04-20-11-29-epoch-12.params',
            'network': 'densenet121',
            'gpu' : 2,
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001

        },
        'neckline_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-10-36-80931/neckline_design_labels-2018-04-20-11-31-epoch-5.params',
            'network': 'densenet121',
            'gpu' : 3,
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001

        },
        'coat_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-10-41-11976/coat_length_labels-2018-04-20-11-32-epoch-8.params',
            'network': 'densenet201',
            'gpu' : 4,
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001

        },
        'neck_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-10-43-33233/neck_design_labels-2018-04-20-11-31-epoch-13.params',
            'network': 'densenet201',
            'gpu' : 5,
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001

        },
        'pant_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-10-44-71409/pant_length_labels-2018-04-20-11-34-epoch-15.params',
            'network': 'densenet121',
            'gpu' : 6,
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001

        },
        'sleeve_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-10-46-14291/sleeve_length_labels-2018-04-20-11-34-epoch-5.params',
            'network': 'densenet201',
            'gpu' : 7,
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001

        }
    }

}
