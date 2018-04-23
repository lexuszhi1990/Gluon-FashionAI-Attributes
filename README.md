### setup:

envs:
```
python : 3.5 or above
pip : 9.1 or above
mxnet : 1.1 or above
```

framework mxnet:
    `pip install mxnet`

install mxnet reference: https://mxnet.incubator.apache.org/install/index.html

### project overview

```
Project
├── data
│   ├── train_valid                       # generated datasets
│   ├── base
│   ├── web
│   ├── rank
│   ├── z_rank
│   └── etc..
├── submit
│   ├── <v1>                             # predicts per categories
│   │   ├── collar_design_labels.csv
│   │   ├── etc...
│   │   └── lapel_design_labels.csv
│   └── submission_uploaded.csv          # final generated results
├── code
│   ├── ckpt                             # pretrained model and ckpt
│   │   ├── densenet201-1cdbc116.params  # densenet201 downloaded for official model zoo
│   │   └── <ckpt_list>
│   ├── logs                             # log files
│   │   └── <log_list>
│   ├── src
│   │   ├── config.py                    # 定义模型名称和路径以及超参数
│   │   ├── symbol.py                    # 模型结构文件
│   │   └── utils.py
│   ├── prepare_train_data.py            # 生成训练图像
│   ├── predict.py                       # 执行测试
│   ├── train.py                         # 执行训练
│   ├── val.py                           # 执行验证
│   └── solver.py                        # 训练/验证/测试 代码
├── README.html
└── README.md
```

### image pre-process

1. make sure that `base` and `web` dataset have been put in dir `data`

or you should change the `root_path` in file: `prepare_train_data.py`
```
root_path = '../data'

label_dir = os.path.join(root_path, 'base', 'Annotations/label.csv')
warmup_label_dir = os.path.join(root_path, 'web', 'Annotations/skirt_length_labels.csv')
```

2. generate the train and validate dataset :`python3 prepare_train_data.py`

it will generate train dataset and val dataset(9:1)
dataset dir is :
![train_val_l2](./code/images/train_val_l2.png)

for one category:
![train_val_l2](./code/images/train_val_l3.png)

### train

#### setup dirs

update `ckpt_path` `training_path` and `validation_path` values in `train.py`. `training_path` and `validation_path` are the path we created in `image pre-process` stage. `ckpt_path` is the path where ckpt to be saved.

```bash
# file: train.py
training_path = "../data/train_valid"
validation_path = "../data/train_valid"
ckpt_path = './ckpt/v1'
```

#### training

update training paremater for tasks in `src/config.py`, for example:

```
'lapel_design_labels' : {
    'model_path': '',
    'network': 'densenet201',
    'loss_type' : 'sfe', # [hinge/sfe]
    'gpus' : [0],
    'num_workers' : 6,
    'batch_size' : 12,
    'lr' : 0.001,
    'wd' : 5e-4,
    'momentum' : 0.9,
    'lr_factor' : 0.75,
    'epochs' : 40,
    'lr_steps' : [10,20,25,30,35,np.inf]
}
```

`sfe` refers to  `SoftmaxCrossEntropy` loss, and `hinge` refers to  `hinge loss`. othter params are straigtforward.

then we can start to train:
`py3 train.py lapel_design_labels`

![train_val_l2](./code/images/train_v2.png)

### testing
#### setup dirs

update `test_dataset_path` and `submission_path` variables depending on custom env in file `predict.py`.

```
# file: predict.py
test_dataset_path = '../data/z_rank'
submission_path = '../submit/v1'
```

update the `mode_path` in `confog.py` with the previous training results.

```
'lapel_design_labels' : {
    'model_path': 'ckpt/v1/lapel_design_labels-2018-04-23-16-37-epoch-2.params',
    'network': 'densenet201',
    ....
}
```

#### test one category

then wen can start to predict single task: `py3 predict.py lapel_design_labels`

![train_val_l2](./code/images/predict_v1.png)

if you update all `modal_path` in `config.py`, you can predict all task by `py3 predict.py`

### post processing

```bash
cd submit/v1
cat *.csv > submisson_20180203_040506.csv
```

### NOTE

- all the code are running under `code` dir.
- the code are tested on Nvidia 1080ti, you should update `batch_size` depend on your machine. set `gpus = []` means training on cpu.

### code references
all the code are available on my github https://github.com/lexuszhi1990/Gluon-FashionAI-Attributes
