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
    }

}