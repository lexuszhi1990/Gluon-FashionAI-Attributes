ps aux | grep python3 | grep collar_design_labels | grep david | awk '{print $2}' | xargs kill -9

ps aux | grep python3 | grep neckline_design_labels | grep david | awk '{print $2}' | xargs kill -9
ps aux | grep python3 | grep skirt_length_labels | grep david | awk '{print $2}' | xargs kill -9
ps aux | grep python3 | grep david
| grep skirt_length_labels | grep david | awk '{print $2}' | xargs kill -9

ps aux | grep python3 | grep david | grep solver.py  | grep coat_length_labels | awk '{print $2}' | xargs kill -9

ps aux | grep python3 | grep david | grep solver.py | grep coat_length_labels | awk '{print $2}' | xargs kill -9

ps aux | grep david | grep python3 | grep train.py | grep coat_length_labels | awk '{print $2}' | xargs kill -9

docker run --rm -it -v /home/david/fashionAI/Gluon-FashionAI-Attributes:/app-dev -v /data/fashion/data/attribute/datasets_david:/mnt/data/attr mxnet-cu90/python:1.2.0-roialign

docker run --rm -it -v /home/david/fashionAI/Gluon-FashionAI-Attributes:/app-dev -v /data/fashion/data/attribute/datasets_david:/mnt/data/attr mxnet/python:1.1.0_gpu_cuda9

python3.5 train_task.py --task collar_design_labels --model resnet50_v2 --num-gpus 4 -j 64 -b 64 --epochs 40
