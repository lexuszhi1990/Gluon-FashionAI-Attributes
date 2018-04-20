from src import utils
from mxnet import gluon, image, init, nd

TRAIN_DATASET_PATH = "/data/david/fai_attr/transfered_data/train_v1"
VAL_DATASET_PATH = "/data/david/fai_attr/transfered_data/val_v1"
test_dataset_path = '/data/david/fai_attr/transfered_data/test_v1'
task_list = ['coat_length_labels', 'lapel_design_labels', 'neckline_design_labels', 'skirt_length_labels', 'collar_design_labels', 'neck_design_labels', 'pant_length_labels', 'sleeve_length_labels']
batch_size = 64
test_data = utils.FaiAttrDataset(test_dataset_path, task_list[0])
test_dataset = gluon.data.DataLoader(test_data, batch_size)
for data, label in test_dataset:
    print(data.shape)
    print(label.shape)
    break


val_data = utils.FaiAttrDataset(VAL_DATASET_PATH, task_list[0])
val_dataset = gluon.data.DataLoader(
    val_data, batch_size)
for data, label in val_dataset:
    print(data.shape)
    print(label.shape)
    break

train_data = utils.FaiAttrDataset(TRAIN_DATASET_PATH, task_list[0])
train_dataset = gluon.data.DataLoader(
    train_data, batch_size, last_batch='discard')
for data, label in train_dataset:
    print(data.shape)
    print(label.shape)
    break


```{.python .input  n=9}
# height x width
input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape)
voc_test = VOCSegDataset(False, input_shape)
```

最后定义批量读取。可以看到跟之前的不同是批量标号不再是一个向量，而是一个三维数组。

```{.python .input  n=10}
batch_size = 64
train_data = gluon.data.DataLoader(
    voc_train, batch_size, shuffle=True,last_batch='discard')
test_data = gluon.data.DataLoader(
    voc_test, batch_size,last_batch='discard')

for data, label in train_data:
    print(data.shape)
    print(label.shape)
    break

coat_length_labels.csv : 11320
collar_design_labels.csv : 8393
lapel_design_labels.csv : 7034
neck_design_labels.csv : 5696
neckline_design_labels.csv : 17148
pant_length_labels.csv : 7460
skirt_length_labels.csv : 19333
sleeve_length_labels.csv : 13299
