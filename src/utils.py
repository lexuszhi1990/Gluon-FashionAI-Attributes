# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon, image, init, nd
import numpy as np
import os, math, argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Gluon for FashionAI Competition',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--solver_type', required=True, type=str,
                        help='Train/Validate/Predict')
    parser.add_argument('--task', required=True, type=str,
                        help='name of the classification task')
    parser.add_argument('--model', required=True, type=str,
                        help='name of the pretrained model from model zoo.')
    parser.add_argument('--model_path', type=str,
                        help='pretrained model path.')
    parser.add_argument('--dataset_path', type=str,
                        help='dataset path.')
    parser.add_argument('-j', '--workers', dest='num_workers', default=64, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--gpus', default="", type=str,
                        help='gpu numbers')
    parser.add_argument('--cpu', help='cpu only', action='store_true')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-factor', default=0.75, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--lr-steps', default='10,20,30', type=str,
                        help='list of learning rate decay epochs as in str')
    args = parser.parse_args()
    return args

def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))

def ten_crop(img, size):
    H, W = size
    iH, iW = img.shape[1:3]

    if iH < H or iW < W:
        raise ValueError('image size is smaller than crop size')

    img_flip = img[:, :, ::-1]
    crops = nd.stack(
        img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img[:, 0:H, 0:W],
        img[:, iH - H:iH, 0:W],
        img[:, 0:H, iW - W:iW],
        img[:, iH - H:iH, iW - W:iW],

        img_flip[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img_flip[:, 0:H, 0:W],
        img_flip[:, iH - H:iH, 0:W],
        img_flip[:, 0:H, iW - W:iW],
        img_flip[:, iH - H:iH, iW - W:iW],
    )
    return (crops)

def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=256,
                                    rand_crop=True, rand_mirror=True,
                                    mean = np.array([0.485, 0.456, 0.406]),
                                    std = np.array([0.229, 0.224, 0.225]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar())

def transform_val(data, label):
    im = data.astype('float32') / 255
    im = image.resize_short(im, 256)
    im, _ = image.center_crop(im, (224, 224))
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return (im, nd.array([label]).asscalar())

def transform_predict(im):
    im = im.astype('float32') / 255
    im = image.resize_short(im, 256)
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    im = ten_crop(im, (224, 224))
    return (im)

def progressbar(i, n, bar_len=40):
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s' % (prog_bar, percents, '%'), end = '\r')

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))


rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def normalize_image(data):
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std

class FaiAttrDataset(gluon.data.Dataset):

    def __init__(self, dataset_path, task):

        self.dataset_path = dataset_path
        self.task = task

        self.label_path = Path(self.dataset_path, "Annotations", "%s.csv"%task)
        assert self.label_path.exists(), "%s not exists" % self.label_path
        self.raw_label, self.data, self.label = self._read_images()

        print('Read '+str(len(self.raw_label))+' examples')

    def _read_images(self):
        lines = self.label_path.open('r').readlines()
        print('load %d images' % len(lines))
        n = len(lines)
        raw_label_list, data, label = [None] * n, [None] * n, [None] * n
        for index, line in enumerate(lines):
            file_name, task, raw_label, output_label_list, output_bbox_list, category_name = line.split(',')
            assert task == self.task, "task is not same"

            raw_img_path = Path(self.dataset_path, file_name)
            assert raw_img_path.exists(), "%s not exists" % raw_img_path
            one_hot_label = [float(i) for i in output_label_list.split('_')]
            try:
                bbox = [float(i) for i in output_bbox_list.split('_') if len(i) > 0]
            except Exception as e:
                pass
            # raw_image = image.imread(raw_img_path.as_posix())
            # data[index] = raw_image
            raw_label_list[index] = {"label": one_hot_label, "label_argmax_index": np.argmax(one_hot_label), "img_path": raw_img_path.as_posix(), "bbox": bbox}
        return raw_label_list, data, label

    def __getitem__(self, idx):
        raw_line = self.raw_label[idx]
        # label = nd.array(raw_line['label'])
        label = nd.array([raw_line['label_argmax_index']])
        raw_image = image.imread(raw_line['img_path'])
        raw_image = image.resize_short(raw_image, 360)
        data = normalize_image(raw_image)
        data = data.transpose((2,0,1))
        return data, label

    def __len__(self):
        return len(self.data)
