from mxnet import gluon, image, init, nd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

PRETRAINED_PATH = './pretrained_model'

def get_pretrained_densenet(model_name, task_num_class, ctx):
    finetune_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True, root=PRETRAINED_PATH)
    with finetune_net.name_scope():
        finetune_net.output = nn.Dense(task_num_class)
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    return finetune_net

def conv_block(channels):
    out = nn.HybridSequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out

def transition_block(channels):
    out = nn.HybridSequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out

class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x


def dense_net(init_channels = 64, growth_rate = 32, block_layers = [6, 12, 24, 16], num_classes = 10):
    net = nn.Sequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7,
                      strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers)-1:
                net.add(transition_block(channels//2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes)
        )
    return net


def get_densenet121_net(num_classes, ctx):
    net = dense_net(init_channels = 64, growth_rate = 32, block_layers = [6, 12, 24, 16], num_classes = num_classes)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net


def get_densenet201_net(num_classes, ctx):
    net = dense_net(init_channels = 64, growth_rate = 32, block_layers = [6, 12, 48, 32], num_classes = num_classes)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net
