# -*- coding: utf-8 -*-

import numpy as np
import os, time, logging, math, argparse
from pathlib import Path

import mxnet as mx
from mxnet import gluon, image, nd
from mxnet import autograd as ag

from src import utils
# available symbols: [densenet121, densenet201, pretrained_densenet121, pretrained_densenet201]
from src.symbol import get_symbol

task_class_num_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9
}

CKPT_PATH = '/data/david/fai_attr/ckpt'
DEFAULT_SUBMISSION_PATH = '/data/david/fai_attr/submissions/default_round2'
DEFAULT_TRAIN_DATASET_PATH = "/data/david/fai_attr/transfered_data/ROUND2/PURE_TRAIN_V1.1"
DEFAULT_VAL_DATASET_PATH = "/data/david/fai_attr/transfered_data/ROUND1/val_v4"

class Solver(object):
    def __init__(self, ckpt_path=None, submission_path=None, validation_path=None, training_path=None):
        self.unique_path_id = "%s-%d" % (time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())), np.random.randint(100000))

        self.training_path = Path(training_path) if training_path else Path(DEFAULT_TRAIN_DATASET_PATH)
        self.validation_path = Path(validation_path) if validation_path else Path(DEFAULT_VAL_DATASET_PATH)
        self.output_submission_path = Path(DEFAULT_SUBMISSION_PATH, self.unique_path_id) if submission_path is None else Path(submission_path)
        self.ckpt_path = Path(CKPT_PATH, self.unique_path_id) if ckpt_path is None else Path(ckpt_path)

    def get_ctx(self):
        return [mx.cpu()] if len(self.gpus) == 0 else [mx.gpu(i) for i in self.gpus]

    def get_image_folder_data(self, dataset_path, task, dataset_type='train'):
        assert dataset_type in ['train', 'val', 'test'], "unknow dataset type %s" % dataset_type
        is_shuffle = True if dataset_type == 'train' else False
        last_batch_type = 'keep' if dataset_type != 'train' else 'discard'

        return gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(Path(dataset_path, task, dataset_type).as_posix(),
                transform=utils.transform_train),
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, last_batch=last_batch_type)

    def get_gluon_dataset(self, dataset_path, task, batch_size, num_workers, dataset_type='train'):
        assert dataset_type in ['train', 'val', 'test'], "unknow dataset type %s" % dataset_type
        is_shuffle = True if dataset_type == 'train' else False
        last_batch_type = 'keep' if dataset_type == 'test' else 'discard'

        fai_dataset = utils.FaiAttrDataset(dataset_path, task, dataset_type=dataset_type)
        return gluon.data.DataLoader(
            fai_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers, last_batch=last_batch_type)

    def predict_cropped_images(self, dataset_path, model_path, task, gpus, network='densenet201', loss_type='sfe'):

        # with Path(dataset_path, 'Annotations/%s.csv' % task).open('r') as f:
        #     self.task_tokens = [l.rstrip().split(',') for l in f.readlines()]
        # self.task_tokens = [t for t in tokens if t[1] == task]

        results_path = self.output_submission_path.joinpath('%s.csv'%(task))
        f_out = results_path.open('w+')
        ctx = self.get_ctx()[0]

        net = get_symbol(network, task_class_num_list[task], ctx)
        net.load_params(model_path, ctx=ctx)
        logging.info("load model from %s" % model_path)

        for index, task_token in enumerate(self.task_tokens):
            img_path, raw_task = task_token[:2]
            assert raw_task == task, "task not match"
            with Path(dataset_path, img_path).open('rb') as f:
                raw_img = f.read()
            img = image.imdecode(raw_img)
            data = utils.transform_cropped_img(img)
            out = net(data.as_in_context(ctx))
            out = nd.SoftmaxActivation(out).mean(axis=0)
            pred_out = ';'.join(["%.8f"%(o) for o in out.asnumpy().tolist()])
            line_out = ','.join([img_path, task, pred_out])
            f_out.write(line_out + '\n')
            utils.progressbar(index, len(self.task_tokens))
        f_out.close()
        logging.info("end predicting for %s, results saved at %s" % (task, results_path))

    def predict_fully_images(self, dataset_path, model_path, task, gpus, network='densenet201', loss_type='sfe'):

        with Path(dataset_path, 'Annotations/%s.csv' % task).open('r') as f:
            task_tokens = [l.rstrip().split(',') for l in f.readlines()]
        results_path = self.output_submission_path.joinpath('%s.csv'%(task))
        f_out = results_path.open('w+')

        ctx = self.get_ctx()[0]
        tic = time.time()
        net = get_symbol(network, task_class_num_list[task], ctx)
        net.load_params(model_path, ctx=ctx)
        logging.info("load model from %s" % model_path)

        for index, task_token in enumerate(task_tokens):
            img_path, raw_task = task_token[:2]
            assert raw_task == task, "task not match %s" % task
            with Path(dataset_path, img_path).open('rb') as f:
                raw_img = f.read()
            img = image.imdecode(raw_img)
            data = utils.transform_fully_img(img)
            out = net(data.as_in_context(ctx))

            if loss_type == 'sfe':
                out = nd.softmax(out)
            elif loss_type == 'hinge':
                out = nd.softmax(out)
            else:
                raise RuntimeError('unknown loss type %s' % loss_type)

            pred_out = ';'.join(["%.8f"%(o) for o in out[0].asnumpy().tolist()])
            line_out = ','.join([img_path, task, pred_out])
            f_out.write(line_out + '\n')
            utils.progressbar(index, len(task_tokens))

        f_out.close()

        logging.info("finish predicting for %s, results are saved at %s | time costs: %.1fs" % (task, results_path, time.time() - tic))

    def predict(self, dataset_path, model_path, task, gpus, network='densenet201', cropped_predict=False, loss_type='sfe'):
        logging.info('starting prediction for %s.' % task)
        self.gpus = gpus

        if not self.output_submission_path.exists():
            self.output_submission_path.mkdir()
        logging.info('submission path: %s' % self.output_submission_path)

        if cropped_predict is True:
            return self.predict_cropped_images(dataset_path, model_path, task, gpus, network, loss_type)
        else:
            return self.predict_fully_images(dataset_path, model_path, task, gpus, network, loss_type)

    def validate(self, symbol, model_path, task, network, gpus, batch_size, num_workers, loss_type='sfe'):
        logging.info('starting validating for %s.' % task)
        tic = time.time()

        self.gpus = gpus
        ctx = self.get_ctx()

        if symbol is None:
            net = get_symbol(network, task_class_num_list[task], ctx)
            net.load_params(model_path, ctx=ctx)
            logging.info("load model from %s" % model_path)
        else:
            net = symbol

        val_data = self.get_gluon_dataset(self.validation_path, task, batch_size, num_workers, dataset_type='val')
        logging.info("load validate dataset from %s" % (self.validation_path))

        metric = mx.metric.Accuracy()
        sfe_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        hinge_loss = gluon.loss.HingeLoss()

        val_loss = 0.0
        pred_correct_count = 0
        pred_count = 0

        for i, batch in enumerate(val_data):

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            hinge_label = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)
            outputs = [net(X) for X in data]

            if loss_type == 'sfe':
                loss = [sfe_loss(yhat, y) for yhat, y in zip(outputs, label)]
            elif loss_type == 'hinge':
                loss = [hinge_loss(yhat, y) for yhat, y in zip(outputs, hinge_label)]
                outputs = [nd.softmax(X) for X in outputs]
            else:
                raise RuntimeError('unknown loss type %s' % loss_type)

            val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            metric.update(label, outputs)
            batch_pred_correct_count, batch_pred_count = utils.calculate_basic_precision(label, outputs)
            pred_count += batch_pred_count
            pred_correct_count += batch_pred_correct_count

            utils.progressbar(i, len(val_data)-1)

        _, val_acc = metric.get()
        val_map, val_loss = pred_correct_count / pred_count, val_loss / len(val_data)
        logging.info('[%s] Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1fs' % (task, val_acc, val_map, val_loss, time.time() - tic))

        return ((val_acc, val_map, val_loss))

    def train(self, task, network, epochs, lr, momentum, wd, lr_factor, lr_steps, gpus, batch_size, num_workers, loss_type='sfe', model_path=None, resume=False):
        logging.info('Entrying the training for Task: %s' % (task))
        self.gpus = gpus
        ctx = self.get_ctx()

        if not self.ckpt_path.exists():
            self.ckpt_path.mkdir()
            logging.info('create ckpt path: %s' % self.ckpt_path)

        train_data = self.get_gluon_dataset(self.training_path, task, batch_size, num_workers, dataset_type='train')
        logging.info("load training dataset from %s" % (self.training_path))

        net = get_symbol(network, task_class_num_list[task], ctx)
        if resume and Path(model_path).exists():
            net.load_params(model_path, ctx=ctx)
            logging.info("load model from %s" % (model_path))

        # Define Trainer
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {
            'learning_rate': lr, 'momentum': momentum, 'wd': wd}, kvstore='device')
        metric = mx.metric.Accuracy()
        sfe_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        hinge_loss = gluon.loss.HingeLoss()
        lr_counter = 0
        num_batch = len(train_data)

        # Start Training
        for epoch in range(epochs):
            if epoch == lr_steps[lr_counter]:
                trainer.set_learning_rate(trainer.learning_rate*lr_factor)
                lr_counter += 1

            tic = time.time()
            train_loss = 0
            metric.reset()
            pred_correct_count = 0
            pred_count = 0

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
                hinge_label = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)
                with ag.record():
                    outputs = [net(X) for X in data]

                    if loss_type == 'sfe':
                        loss = [sfe_loss(yhat, y) for yhat, y in zip(outputs, label)]
                    elif loss_type == 'hinge':
                        loss = [hinge_loss(yhat, y) for yhat, y in zip(outputs, hinge_label)]
                        outputs = [nd.softmax(X) for X in outputs]
                    else:
                        raise RuntimeError('unknown loss type %s' % loss_type)
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                metric.update(label, outputs)

                # ap, cnt = utils.calculate_ap(label, outputs)
                batch_pred_correct_count, batch_pred_count = utils.calculate_basic_precision(label, outputs)
                pred_count += batch_pred_count
                pred_correct_count += batch_pred_correct_count

                utils.progressbar(i, num_batch-1)

            train_map = pred_correct_count / pred_count
            _, train_acc = metric.get()
            train_loss /= num_batch

            saved_path = self.ckpt_path.joinpath('%s-%s-epoch-%d.params' % (task, time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())), epoch))
            net.save_params(saved_path.as_posix())
            logging.info('\nsave results at %s' % saved_path)
            val_acc, val_map, val_loss = self.validate(net, model_path=None, task=task, network=network, gpus=gpus, batch_size=batch_size, num_workers=num_workers, loss_type=loss_type)
            # val_acc, val_map, val_loss = 0, 0, 0
            logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1fs' %
                     (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))

        return net
