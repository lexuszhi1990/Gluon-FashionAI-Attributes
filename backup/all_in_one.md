
### generate coco-like dataset

docker run -it --rm -p 9999:8888 -v /home/david/fashionAI/build-datasets-from-zero:/notebooks -v /data/david/cocoapi:/mnt/cocoapi -v /data/david/fai_attr:/mnt/data/fai_attr registry.cn-shenzhen.aliyuncs.com/deeplearn/jupyter-py3 bash

SOURCE_DIR='/mnt/data/fai_attr/raw_data/val_v1'
DEST_DIR='/mnt/data/fai_attr/raw_data/val_v1'
IMAGE_SET='question'

bash fai_attr_generator.sh
`/mnt/data/fai_attr/raw_data/val_v1/Tests/question.json`

### generate bbox

docker run --rm -it -v /home/david/fashionAI/mx-rcnn:/app-dev -v /data/david/fai_attr:/mnt/data/fai_attr -v /data/david/models/fai:/mnt/models -v /data/david/cocoapi:/mnt/coco mxnet-cu90/python:1.2.0-roialign

bash scripts/vgg_fashion_rcnn.sh

### generate the results
