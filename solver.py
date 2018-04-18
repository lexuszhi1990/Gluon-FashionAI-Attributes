import mxnet as mx
import numpy as np
import os, time, logging, math, argparse
from pathlib import Path

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
SUBMISSION_PATH = '/data/david/fai_attr/gloun_data/submission'

class ValSolver(object):
    def __init__(self, batch_size=32, num_workers=16, num_gpu=None):
        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_path = Path(DATASET_PATH)
        self.submission_path = Path(SUBMISSION_PATH)
        self.output_path = Path(self.submission_path, time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time())))

        lines = Path(self.submission_path, 'Tests/question.csv').open('r').readlines()
        self.tokens = [l.rstrip().split(',') for l in lines]

    def predict(self, model_path, task, network='densenet201'):
        logging.info('starting prediction for %s.\n' % task)

        if not self.output_path.exists():
            self.output_path.mkdir()

        task_tokens = [t for t in self.tokens if t[1] == task]

        results_path = self.output_path.joinpath('%s.csv'%(task))
        f_out = results_path.open('w+')
        ctx = mx.gpu(self.num_gpu) if self.num_gpu else mx.cpu()

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

    def get_validate_data(self, task):
        val_data = gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(self.dataset_path.joinpath(task, 'val').as_posix(),
                transform=utils.transform_val),
            batch_size=self.batch_size, shuffle=False, num_workers =self.num_workers)

        return val_data

    def validate(self, model_path, task, network='densenet201'):
        logging.info('starting validating for %s.\n' % task)

        ctx = mx.gpu(self.num_gpu) if self.num_gpu else mx.cpu()
        net = get_pretrained_model(network, task_list[task], ctx)
        net.load_params(model_path, ctx=ctx)
        logging.info("load model from %s" % model_path)

        val_data = self.get_validate_data(task)
        logging.info("load validate dataset from %s for %s" % (self.dataset_path, task))

        metric = mx.metric.Accuracy()
        L = gluon.loss.SoftmaxCrossEntropyLoss()
        AP = 0.
        AP_cnt = 0
        val_loss = 0
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[ctx], batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[ctx], batch_axis=0, even_split=False)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            ap, cnt = utils.calculate_ap(label, outputs)
            AP += ap
            AP_cnt += cnt
        val_acc, val_map, val_loss = metric.get(), AP / AP_cnt, val_loss / len(val_data)
        logging.info('[%s] Val-acc: %sf, mAP: %.3f, loss: %.3f' % (task, val_acc[1], val_map, val_loss))

if __name__ == "__main__":
    solver = ValSolver(num_gpu=7)
    # task list:
    # 'collar_design_labels',
    # 'skirt_length_labels',
    # 'lapel_design_labels',
    # 'neckline_design_labels',
    # 'coat_length_labels',
    # 'neck_design_labels',
    # 'pant_length_labels',
    # 'sleeve_length_labels',
    # solver.predict(model_path='/data/david/models/fai_attrbutes/v1/sleeve_length_labels-2018-04-18-14-46-epoch-37.params', task='sleeve_length_labels')
    solver.validate(model_path='/data/david/models/fai_attrbutes/v1/sleeve_length_labels-2018-04-18-14-46-epoch-37.params', task='sleeve_length_labels')

