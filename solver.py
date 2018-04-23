# -*- coding: utf-8 -*-

import numpy as np
import os, time, logging, math, argparse
from pathlib import Path

import mxnet as mx
from mxnet import gluon, image, nd
from mxnet import autograd as ag

from src import utils
from src.symbol import get_pretrained_model

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

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

CKPT_PATH = '/data/david/fai_attr/gloun_data/ckpt'
DEFAULT_SUBMISSION_PATH = '/data/david/fai_attr/gloun_data/submission'
DEFAULT_TRAIN_DATASET_PATH = "/data/david/fai_attr/transfered_data/train_v1"
DEFAULT_VAL_DATASET_PATH = "/data/david/fai_attr/transfered_data/val_v1"

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


    def get_gluon_dataset(self, dataset_path, task, dataset_type='train'):
        assert dataset_type in ['train', 'val', 'test'], "unknow dataset type %s" % dataset_type

        is_shuffle = True if dataset_type == 'train' else False
        last_batch_type = 'keep' if dataset_type != 'train' else 'discard'

        fai_dataset = utils.FaiAttrDataset(dataset_path, task, dataset_type=dataset_type)
        return gluon.data.DataLoader(
            fai_dataset, batch_size=self.batch_size, shuffle=is_shuffle, num_workers=self.num_workers, last_batch=last_batch_type)

    def predict_cropped_images(self, dataset_path, model_path, task, gpus, network='densenet201'):
        results_path = self.output_submission_path.joinpath('%s.csv'%(task))
        f_out = results_path.open('w+')
        ctx = self.get_ctx()[0]

        net = get_pretrained_model(network, task_class_num_list[task], ctx)
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

    def predict_fully_images(self, dataset_path, model_path, task, gpus, network='densenet201'):
        results_path = self.output_submission_path.joinpath('%s.csv'%(task))
        f_out = results_path.open('w+')
        ctx = self.get_ctx()[0]

        net = get_pretrained_model(network, task_class_num_list[task], ctx)
        net.load_params(model_path, ctx=ctx)
        logging.info("load model from %s" % model_path)
        for index, task_token in enumerate(task_tokens):
            img_path, raw_task = task_token[:2]
            assert raw_task == task, "task not match"
            with Path(dataset_path, img_path).open('rb') as f:
                raw_img = f.read()
            img = image.imdecode(raw_img)
            data = utils.transform_fully_img(img)
            out = net(data.as_in_context(ctx))
            out = nd.softmax(out)
            pred_out = ';'.join(["%.8f"%(o) for o in out[0].asnumpy().tolist()])
            line_out = ','.join([path, task, pred_out])
            f_out.write(line_out + '\n')
            utils.progressbar(index, len(task_tokens))
        f_out.close()
        logging.info("finish predicting for %s, results are saved at %s" % (task, results_path))

    def predict(self, dataset_path, model_path, task, gpus, network='densenet201', cropped_predict=True):
        logging.info('starting prediction for %s.' % task)
        self.gpus = gpus
        if not self.output_submission_path.exists():
            self.output_submission_path.mkdir()
        logging.info('submission path: %s' % self.output_submission_path)
        with Path(dataset_path, 'Tests/question.csv').open('r') as f:
            tokens = [l.rstrip().split(',') for l in f.readlines()]
        self.task_tokens = [t for t in tokens if t[1] == task]

        if cropped_predict is True:
            return self.predict_cropped_images(dataset_path, model_path, task, gpus, network)
        else:
            return self.predict_fully_images(dataset_path, model_path, task, gpus, network)

    def validate(self, symbol, model_path, task, network, gpus, batch_size, num_workers):
        logging.info('starting validating for %s.' % task)
        self.gpus = gpus
        ctx = self.get_ctx()

        self.num_workers = num_workers
        self.batch_size = batch_size

        if symbol is None:
            net = get_pretrained_model(network, task_class_num_list[task], ctx)
            net.load_params(model_path, ctx=ctx)
            logging.info("load model from %s" % model_path)
        else:
            net = symbol

        val_data = self.get_image_folder_data(self.training_path ,task, dataset_type='train')
        logging.info("load validate dataset from %s" % (self.validation_path))

        metric = mx.metric.Accuracy()
        L = gluon.loss.SoftmaxCrossEntropyLoss()
        AP = 0.
        AP_cnt = 0
        val_loss = 0
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            ap, cnt = utils.calculate_ap(label, outputs)
            AP += ap
            AP_cnt += cnt
        _, val_acc = metric.get()
        val_map, val_loss = AP / AP_cnt, val_loss / len(val_data)
        logging.info('[%s] Val-acc: %.3f, mAP: %.3f, loss: %.3f' % (task, val_acc, val_map, val_loss))
        return ((val_acc, val_map, val_loss))

    def train(self, task, model_name, epochs, lr, momentum, wd, lr_factor, lr_steps, gpus, batch_size, num_workers, loss_type='sfe'):

        self.gpus = gpus
        ctx = self.get_ctx()

        self.num_workers = num_workers
        self.batch_size = batch_size * max(len(self.gpus), 1)
        logging.info('Start Training for Task: %s' % (task))

        if not self.ckpt_path.exists():
            self.ckpt_path.mkdir()
            logging.info('create ckpt path: %s' % self.ckpt_path)

        finetune_net = get_pretrained_model(model_name, task_class_num_list[task], ctx)

        train_data = self.get_image_folder_data(self.training_path ,task, dataset_type='train')
        logging.info("load training dataset from %s" % (self.training_path))

        # Define Trainer
        trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
            'learning_rate': lr, 'momentum': momentum, 'wd': wd})
        metric = mx.metric.Accuracy()
        sfe_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        hing_loss = gluon.loss.HingeLoss()
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
            AP = 0.
            AP_cnt = 0

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
                argmax_index_label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
                hinge_label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
                with ag.record():
                    outputs = [finetune_net(X) for X in data]

                    if loss_type == 'sfe':
                        label = argmax_index_label
                        loss = [sfe_loss(yhat, y) for yhat, y in zip(outputs, label)]
                    elif loss_type == 'hinge':
                        label = hinge_label
                        loss = [hing_loss(yhat, y) for yhat, y in zip(outputs, label)]
                    else:
                        raise RuntimeError('un avaliable loss type %s' % loss_type)
                for l in loss:
                    l.backward()
                trainer.step(self.batch_size)
                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

                metric.update(label, outputs)

                if loss_type == 'sfe':
                    ap, cnt = utils.calculate_ap(label, outputs)
                elif loss_type == 'hinge':
                    ap, cnt = utils.calculate_ap_full(label, outputs)
                else:
                    raise RuntimeError('un avaliable loss type %s' % loss_type)

                AP += ap
                AP_cnt += cnt

                utils.progressbar(i, num_batch-1)

            train_map = AP / AP_cnt
            _, train_acc = metric.get()
            train_loss /= num_batch

            saved_path = self.ckpt_path.joinpath('%s-%s-epoch-%d.params' % (task, time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())), epoch))
            finetune_net.save_params(saved_path.as_posix())

            val_acc, val_map, val_loss = self.validate(finetune_net, model_path=None, task=task, network=model_name, gpus=gpus, batch_size=self.batch_size, num_workers=self.num_workers)
            # val_acc, val_map, val_loss = 0, 0, 0

            logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1fs' %
                     (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))
            logging.info('\nsave results at %s' % saved_path)

        return finetune_net
