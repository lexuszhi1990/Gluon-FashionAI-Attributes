# -*- coding: utf-8 -*-

import time
from pathlib import Path
from solver import Solver

# task list:
# 'collar_design_labels',
# 'skirt_length_labels',
# 'lapel_design_labels',
# 'neckline_design_labels',
# 'coat_length_labels',
# 'neck_design_labels',
# 'pant_length_labels',
# 'sleeve_length_labels',

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

solver = Solver(batch_size=24, num_workers=16, gpus=[7], solver_type='Validate')


results_file_path = Path('./results_roadmap.md')
f_out = results_file_path.open('a')
f_out.write('%s :\n' % time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())))

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
