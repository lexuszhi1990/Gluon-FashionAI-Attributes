### setup:

envs:
```
python : 3.5 or above
pip : 9.1 or above
mxnet : 1.1 or above
```

framework mxnet:
    pip install mxnet

install mxnet reference: https://mxnet.incubator.apache.org/install/index.html

### project overview

```
Project
├── data
│   ├── base
│   ├── rank
│   ├── z_rank
│   └── etc..
├── submit
│   ├── v1                              # submission per task
│   │   ├── collar_design_labels.csv
│   │   ├── etc...
│   │   └── submission_v1.csv
│   └── submission_upload.csv
├── code
│   ├── ckpt
│   │   └── <trained_modes_list>
│   ├── logs
│   ├── src
│   │   ├── config.py
│   │   ├── symbol.py
│   │   └── utils.py
│   ├── predict.py
│   └── solver.py
└── README.md
```

### usage

update `test_dataset_path` and `submission_path` variables depending on custom env in file `predict.py`.

```
test_dataset_path = '../data/z_rank'
submission_path = '../submit/v1'
```

predict single task: `py3 predict.py lapel_design_labels`
outputs:
```
start predict single task: lapel_design_labels
INFO:root:starting prediction for lapel_design_labels.
INFO:root:load model from /data/david/models/fai_attrbutes/v1/lapel_design_labels-2018-04-18-16-59-epoch-39.params
INFO:root:end predicting for lapel_design_labels, results saved at ../submit/v1/lapel_design_labels.csv
```

predict all task: `py3 predict.py`

### post precessing

```bash
cd submit/v1
cat *.csv > submisson_20180203_040506.csv
```

### results path:

V1: /data/david/fai_attr/gloun_data/submission/2018-04-19-09-33-77167
V2: /data/david/fai_attr/gloun_data/submission/2018-04-20-20-49-32605
