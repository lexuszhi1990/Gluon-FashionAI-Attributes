# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import gluon, image, init, nd
import numpy as np
import os, math, argparse
from pathlib import Path
import time
import logging

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])
BASE_SHAPE = 360

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

def setup_log(log_id):
    log_path = 'logs/%s-%s.log' % (log_id, "%s"%(time.strftime("%Y-%m-%d-%H-%M")))
    logging.basicConfig(level=logging.INFO,
                        handlers = [
                            logging.StreamHandler(),
                            logging.FileHandler(log_path)
                        ])
    logging.info('create log file: %s' % log_path)

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

def calculate_ap_full(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = np.argmax(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))

# https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.28013a26P0p2gQ&raceId=231649
# 我们还会展示BasicPrecision指标，即模型在测试集全部预测输出(ProbThreshold=0)情况下每个属性维度准确率的平均值，
# 作为更直接的准确率预估指标供大家参考。在BasicPrecision = 0.7时，排名得分mAP一般在 0.93 左右
def calculate_basic_precision(labels, outputs, prob_threshold=0):
    pred_count = 0
    pred_correct_count = 0
    pred_outputs = [nd.softmax(X) for X in outputs]
    for label, output in zip(labels, pred_outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            if np.max(op) >= prob_threshold:
                if np.argmax(op) == np.argmax(lb):
                    pred_correct_count += 1
                pred_count += 1
    return ((pred_correct_count, pred_count))

def normalize_image(data):
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std

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

def transform_cropped_img(im):
    im = im.astype('float32') / 255
    im = image.resize_short(im, 256)
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    im = ten_crop(im, (224, 224))
    return (im)

def transform_fully_img(im):
    im = image.resize_short(im, BASE_SHAPE)
    im = normalize_image(im)
    im = im.transpose((2,0,1))
    im = im.expand_dims(axis=0)
    return (im)

def progressbar(i, n, bar_len=40):
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s' % (prog_bar, percents, '%'), end = '\r')

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

class FaiAttrDataset(gluon.data.Dataset):

    def __init__(self, dataset_path, task, dataset_type='train'):

        self.dataset_path = dataset_path
        self.task = task
        self.dataset_type = dataset_type

        self.label_path = Path(self.dataset_path, "Annotations", "%s.csv"%task)
        assert self.label_path.exists(), "%s not exists" % self.label_path
        self.raw_label = self._read_images()


    def _convert_label_to_one_hot(self, raw_label):
        label_y = [int(l == 'y')*1 for l in raw_label]
        label_m = [int(l == 'm')*0.5 for l in raw_label]
        label = [x+y for x,y in zip(label_m, label_y)]
        return label

    def _read_images(self):
        lines = self.label_path.open('r').readlines()
        print('load %d images' % len(lines))
        n = len(lines)
        raw_label_list = [None] * n
        for index, line in enumerate(lines):
            file_name, task, raw_fai_label, output_label_list, output_bbox_list, category_name = line.split(',')
            assert task == self.task, "task is not same"

            raw_img_path = Path(self.dataset_path, file_name)
            assert raw_img_path.exists(), "%s not exists" % raw_img_path
            # one_hot_label = [float(i) for i in output_label_list.split('_')]
            one_hot_label = self._convert_label_to_one_hot(raw_fai_label)
            hinge_label = []
            for l in one_hot_label:
                hinge_label.append(l if l > 0 else -1)
            bbox = [int(i) for i in output_bbox_list.split('_') if len(i) > 0]

            # img_path = raw_img_path.as_posix()
            # raw_image = image.imread(img_path)

            raw_label_list[index] = {"one_hot_label": one_hot_label, "argmax_index_label": np.argmax(one_hot_label), "hinge_label": hinge_label, "img_path": raw_img_path.as_posix(), "bbox": bbox}
        return raw_label_list

    def get_img(self):
        img_path = raw_img_path.as_posix()
        raw_image = image.imread(img_path)
        raw_mask = nd.zeros((raw_image.shape[0], raw_image.shape[1], 1)).astype(np.uint8)
        concated_data = nd.zeros((raw_image.shape[0], raw_image.shape[1], 4))
        raw_mask[bbox[0]:width, bbox[1]:height] = 255
        mask_raw_img = nd.concat(raw_image, raw_mask, dim=2)
        norm_mask_raw_img = normalize_image(mask_raw_img)
        resized_norm_mask_raw_img = image.resize_short(norm_mask_raw_img, BASE_SHAPE)
        data = resized_norm_mask_raw_img.transpose((2,0,1))

    def __getitem__(self, idx):
        raw_line = self.raw_label[idx]
        img_path, bbox = raw_line['img_path'], raw_line['bbox']
        raw_image = image.imread(img_path)
        raw_image = image.resize_short(raw_image, BASE_SHAPE)
        if self.dataset_type == 'train':
            raw_image = image.HorizontalFlipAug(0.5)(raw_image)
        raw_image = normalize_image(raw_image)
        data = raw_image.transpose((2,0,1))
        return data, nd.array([raw_line['argmax_index_label']]), nd.array(raw_line['hinge_label'])

    def __len__(self):
        return len(self.raw_label)
