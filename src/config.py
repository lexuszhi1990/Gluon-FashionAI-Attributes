# -*- coding: utf-8 -*-
# bz: 12 -> 10G
# bz: 10 -> 9G
# bz: 8 -> 8G
# 172 gpu:

# .vvv....
# v.vv....
# vv.v....
# vvv.....
# .....vvv
# ....v.vv
# ....vv.v
# ....vvv.


from easydict import EasyDict as edict
import numpy as np

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
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-18-15-58023/collar_design_labels-2018-04-20-19-39-epoch-17.params',
            'network': 'densenet201',
            'gpus' : [0],
            'num_workers' : 2,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 20,
            'lr_steps' : [5,8,10,15,np.inf]
        },
        'skirt_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-16-51-91422/skirt_length_labels-2018-04-20-19-43-epoch-15.params',
            'network': 'densenet201',
            'gpus' : [1],
            'num_workers' : 2,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 20,
            'lr_steps' : [5,8,10,15,np.inf]
        },
        'lapel_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-16-56-97706/lapel_design_labels-2018-04-20-18-37-epoch-18.params',
            'network': 'densenet121',
            'gpus' : [2],
            'num_workers' : 2,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 20,
            'lr_steps' : [5,8,10,15,np.inf]
        },
        'neckline_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-16-56-63802/neckline_design_labels-2018-04-20-20-00-epoch-19.params',
            'network': 'densenet121',
            'gpus' : [3],
            'num_workers' : 2,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 20,
            'lr_steps' : [5,8,10,15,np.inf]
        },
        'coat_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-18-10-3387/coat_length_labels-2018-04-20-20-40-epoch-19.params',
            'network': 'densenet201',
            'gpus' : [4],
            'num_workers' : 2,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 20,
            'lr_steps' : [5,8,10,15,np.inf]
        },
        'neck_design_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-16-58-62615/neck_design_labels-2018-04-20-19-05-epoch-19.params',
            'network': 'densenet201',
            'gpus' : [5],
            'num_workers' : 2,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 20,
            'lr_steps' : [5,8,10,15,np.inf]

        },
        'pant_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-17-35-93141/pant_length_labels-2018-04-20-18-54-epoch-19.params',
            'network': 'densenet121',
            'gpus' : [6],
            'num_workers' : 4,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 20,
            'lr_steps' : [5,8,10,15,np.inf]

        },
        'sleeve_length_labels' : {
            'model_path': '/data/david/fai_attr/gloun_data/ckpt/2018-04-20-18-12-4240/sleeve_length_labels-2018-04-20-19-44-epoch-13.params',
            'network': 'densenet201',
            'gpus' : [7],
            'num_workers' : 4,
            'batch_size' : 12,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 20,
            'lr_steps' : [5,8,10,15,np.inf]

        }
    },

    "v4": {
        'collar_design_labels' : {
            'model_path': '/data/david/fai_attr/ckpt/2018-05-03-18-22-28163/collar_design_labels-2018-05-03-19-44-epoch-39.params',
            'network': 'pretrained_densenet201',
            'loss_type' : 'hinge', # [hinge/sfe]
            'gpus' : [0, 1, 2, 3],
            'num_workers' : 24,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 40,
            'lr_steps' : [10,25,30,35,np.inf]
        },
        'skirt_length_labels' : {
            'model_path': '/data/david/fai_attr/ckpt/2018-05-04-10-52-19712/skirt_length_labels-2018-05-04-13-09-epoch-39.params',
            'network': 'pretrained_densenet201',
            'loss_type' : 'hinge', # [hinge/sfe]
            'gpus' : [4,5],
            'num_workers' : 12,
            'batch_size' : 8,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 40,
            'lr_steps' : [10,25,30,35,np.inf]
        },
        'lapel_design_labels' : {
            'model_path': '/data/david/fai_attr/ckpt/2018-05-03-19-50-81092/lapel_design_labels-2018-05-03-20-48-epoch-18.params',
            'network': 'pretrained_densenet201',
            'loss_type' : 'hinge', # [hinge/sfe]
            'gpus' : [2, 3],
            'num_workers' : 6,
            'batch_size' : 10,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 40,
            'lr_steps' : [10,25,30,35,np.inf]
        },
        'neckline_design_labels' : {
            'model_path': '/data/david/fai_attr/ckpt/2018-05-03-18-31-83724/neckline_design_labels-2018-05-03-20-51-epoch-16.params',
            'network': 'pretrained_densenet201',
            'loss_type' : 'hinge', # [hinge/sfe]
            'gpus' : [5],
            'num_workers' : 6,
            'batch_size' : 12,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 40,
            'lr_steps' : [10,25,30,35,np.inf]
        },
        'coat_length_labels' : {
            'model_path': '/data/david/fai_attr/ckpt/2018-05-04-16-49-50624/coat_length_labels-2018-05-04-17-52-epoch-15.params',
            'network': 'pretrained_densenet201',
            'loss_type' : 'hinge', # [hinge/sfe]
            'gpus' : [0,1],
            'num_workers' : 6,
            'batch_size' : 10,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 40,
            'lr_steps' : [10,25,30,35,np.inf]
        },
        'neck_design_labels' : {
            'model_path': '/data/david/fai_attr/ckpt/2018-05-03-17-57-48502/neck_design_labels-2018-05-03-19-32-epoch-39.params',
            'network': 'pretrained_densenet201',
            'loss_type' : 'sfe', # [hinge/sfe]
            'gpus' : [4,5],
            'num_workers' : 12,
            'batch_size' : 12,
            'lr' : 0.001,
            'wd' : 1e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 40,
            'lr_steps' : [10,25,30,35,np.inf]

        },
        'pant_length_labels' : {
            'model_path': '/data/david/fai_attr/ckpt/2018-05-04-11-07-29483/pant_length_labels-2018-05-04-13-37-epoch-39.params',
            'network': 'pretrained_densenet201',
            'loss_type' : 'hinge', # [hinge/sfe]
            'gpus' : [4,5],
            'num_workers' : 12,
            'batch_size' : 12,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 40,
            'lr_steps' : [10,25,30,35,np.inf]

        },
        'sleeve_length_labels' : {
            'model_path': '/data/david/fai_attr/ckpt/2018-05-04-16-27-14786/sleeve_length_labels-2018-05-04-17-54-epoch-17.params',
            'network': 'pretrained_densenet201',
            'loss_type' : 'hinge', # [hinge/sfe]
            'gpus' : [2,3],
            'num_workers' : 12,
            'batch_size' : 12,
            'lr' : 0.001,
            'wd' : 5e-4,
            'momentum' : 0.9,
            'lr_factor' : 0.75,
            'epochs' : 40,
            'lr_steps' : [10,25,30,35,np.inf]

        }
    }

}
