# -*- coding: utf-8 -*-

# USAGE:
# python3 solver.py --solver_type Train --model densenet201 --task coat_length_labels --gpu 7 --b 24 -j 16
# python3 solver.py --solver_type Validate --model densenet201 --model_path /data/david/models/fai_attrbutes/v1/sleeve_length_labels-2018-04-18-14-50-epoch-39.params --task sleeve_length_labels --gpu 7 --b 24 -j 16
# python3 solver.py --solver_type Predict --model densenet201 --model_path /data/david/models/fai_attrbutes/v1/sleeve_length_labels-2018-04-18-14-50-epoch-39.params --task sleeve_length_labels --gpu 7 --b 24 -j 16


import numpy as np
import os, time, logging, math, argparse
from pathlib import Path

import mxnet as mx
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

from src import utils
from src.symbol import get_pretrained_model

logging.basicConfig(level=logging.INFO,
                    handlers = [
                        logging.StreamHandler(),
                        logging.FileHandler('training.log')
                    ])

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

task_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9
}

DATASET_PATH = '/data/david/fai_attr/gloun_data/train_valid'
CKPT_PATH = '/data/david/fai_attr/gloun_data/ckpt'
SUBMISSION_PATH = '/data/david/fai_attr/gloun_data/submission'

class Solver(object):
    def __init__(self, batch_size=None, num_workers=None, gpus=None, cpu=None, solver_type='Train'):
        self.solver_type = solver_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gpus = gpus
        self.cpu = cpu

        self.dataset_path = Path(DATASET_PATH)
        self.submission_path = Path(SUBMISSION_PATH)
        self.ckpt_path = Path(CKPT_PATH)

        path_id = "%s-%d" % (time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())), np.random.randint(100000))
        self.output_submission_path = Path(self.submission_path, path_id)
        self.output_ckpt_path = Path(self.ckpt_path, path_id)

        if self.solver_type != 'Train':
            lines = Path(self.submission_path, 'Tests/question.csv').open('r').readlines()
            self.tokens = [l.rstrip().split(',') for l in lines]

    def get_ctx(self):
        return [mx.cpu()] if self.cpu else [mx.gpu(i) for i in self.gpus]

    def get_validate_data(self, task):
        return gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(self.dataset_path.joinpath(task, 'val').as_posix(),
                transform=utils.transform_val),
            batch_size=self.batch_size, shuffle=False, num_workers =self.num_workers)

    def get_train_data(self, task):
        return gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(self.dataset_path.joinpath(task, 'val').as_posix(),
                transform=utils.transform_train),
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, last_batch='discard')

    def validate(self, symbol, model_path, task, network='densenet201'):
        logging.info('starting validating for %s.\n' % task)

        ctx = self.get_ctx()
        if symbol is None:
            net = get_pretrained_model(network, task_list[task], ctx)
            net.load_params(model_path, ctx=ctx)
            logging.info("load model from %s" % model_path)
        else:
            net = symbol

        val_data = self.get_validate_data(task)
        logging.info("load validate dataset from %s" % (self.dataset_path))

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

    def predict(self, model_path, task, network='densenet201'):
        logging.info('starting prediction for %s.\n' % task)

        if not self.output_submission_path.exists():
            self.output_submission_path.mkdir()
            logging.info('create %s' % self.output_submission_path)

        task_tokens = [t for t in self.tokens if t[1] == task]

        results_path = self.output_submission_path.joinpath('%s.csv'%(task))
        f_out = results_path.open('w+')
        ctx = self.get_ctx()[0]

        net = get_pretrained_model(network, task_list[task], ctx)
        net.load_params(model_path, ctx=ctx)
        logging.info("load model from %s" % model_path)

        for index, (path, task, _) in enumerate(task_tokens):
            raw_img = Path(self.submission_path, path).open('rb').read()
            img = image.imdecode(raw_img)
            data = utils.transform_predict(img)
            out = net(data.as_in_context(ctx))
            out = nd.SoftmaxActivation(out).mean(axis=0)
            pred_out = ';'.join(["%.8f"%(o) for o in out.asnumpy().tolist()])
            line_out = ','.join([path, task, pred_out])
            f_out.write(line_out + '\n')
            utils.progressbar(index, len(task_tokens))
        f_out.close()
        logging.info("end predicting for %s, results saved at %s" % (task, results_path))

    def train(self, args):
        task = args.task
        model_name = args.model
        epochs = args.epochs
        lr = args.lr
        batch_size = args.batch_size
        momentum = args.momentum
        wd = args.wd
        lr_factor = args.lr_factor
        lr_steps = [int(s) for s in args.lr_steps.split(',')] + [np.inf]
        num_workers = args.num_workers
        ctx = self.get_ctx()
        batch_size = batch_size * max(len(self.gpus), 1)

        logging.info('Start Training for Task: %s\n' % (task))

        if not self.output_ckpt_path.exists():
            self.output_ckpt_path.mkdir()
            logging.info('create %s' % self.output_ckpt_path)

        finetune_net = get_pretrained_model(model_name, task_list[task], ctx)

        train_data = self.get_train_data(task)
        logging.info("load train dataset from %s for %s" % (self.dataset_path, task))

        # Define Trainer
        trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
            'learning_rate': lr, 'momentum': momentum, 'wd': wd})
        metric = mx.metric.Accuracy()
        L = gluon.loss.SoftmaxCrossEntropyLoss()
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
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
                with ag.record():
                    outputs = [finetune_net(X) for X in data]
                    loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
                for l in loss:
                    l.backward()

                trainer.step(batch_size)
                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

                metric.update(label, outputs)
                ap, cnt = utils.calculate_ap(label, outputs)
                AP += ap
                AP_cnt += cnt

                utils.progressbar(i, num_batch-1)

            train_map = AP / AP_cnt
            _, train_acc = metric.get()
            train_loss /= num_batch


            saved_path = self.output_ckpt_path.joinpath('%s-%s-epoch-%d.params' % (task, time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())), epoch))
            finetune_net.save_params(saved_path.as_posix())
            logging.info('\nsave results at %s' % saved_path)

            val_acc, val_map, val_loss = self.validate(finetune_net, model_path=None, task=task, network=model_name)

            logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1fs' %
                     (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))

        return finetune_net

if __name__ == "__main__":
    args = utils.parse_args()
    args.gpus = [int(i) for i in args.gpus.split(',') if len(args.gpus) > 0]
    solver = Solver(batch_size=args.batch_size, num_workers=args.num_workers, gpus=args.gpus, cpu=args.cpu, solver_type=args.solver_type)

    if args.solver_type == "Train":
        solver.train(args)
    elif args.solver_type == "Validate":
        # validate(self, symbol, model_path, task, network)
        solver.validate(None, model_path=args.model_path, task=args.task, network=args.model)
    elif args.solver_type == "Predict":
        # predict(self, model_path, task, network):
        solver.predict(model_path=args.model_path, task=args.task, network=args.model)
