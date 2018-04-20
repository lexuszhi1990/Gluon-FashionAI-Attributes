from mxnet import gluon, image, init, nd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

PRETRAINED_PATH = './pretrained_model'
# model_name='densenet201'
model_name='densenet121'
task_num_class = 10

finetune_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True, root=PRETRAINED_PATH)

(finetune_net.features[:-1], finetune_net.output)

with finetune_net.name_scope():
    finetune_net.output = nn.Dense(task_num_class)
finetune_net.output.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()

net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)



net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)

x = nd.random.uniform(shape=(1,3,*input_shape))
print('Input:', x.shape)
print('Output:', net(x).shape)



net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)

x = nd.random.uniform(shape=(1,3,*input_shape))
print('Input:', x.shape)
print('Output:', net(x).shape)
```


```{.python .input  n=14}
num_classes = len(classes)

with net.name_scope():
    net.add(
        nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16,strides=32)
    )
