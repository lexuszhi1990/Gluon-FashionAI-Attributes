from mxnet import gluon, image, init, nd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

PRETRAINED_PATH = './pretrained_model'

def get_pretrained_model(model_name, task_num_class, ctx):
    finetune_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True, root=PRETRAINED_PATH)
    with finetune_net.name_scope():
        finetune_net.output = nn.Dense(task_num_class)
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    return finetune_net
