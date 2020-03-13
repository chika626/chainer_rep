import argparse
import copy
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import crate_dataset as c_d

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

import cv2
cv2.cv2.setNumThreads(0)



class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def forward(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label


def main():
    # cuDNNのautotuneを有効にする
    chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
    chainer.config.autotune = True
    chainer.config.cv_resize_backend = "cv2"

    gpu_id = 0
    batchsize = 8
    out_num = 'results'
    log_interval = 1, 'epoch'
    epoch_max = 100
    initial_lr = 0.0001
    lr_decay_rate = 0.1
    lr_decay_timing = [20, 40, 60, 80]

    majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"

    # モデルの設定
    model = SSD512(n_fg_class=len(majomoji_label), pretrained_model='imagenet')
    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)

    # GPUの設定
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()

    # データセットの設定
    # train_dataset = ConcatenatedDataset(
    #         VOCBboxDataset(year='2007', split='trainval'),
    #         VOCBboxDataset(year='2012', split='trainval'))


    # train_dataset=VOCBboxDataset(year='2007', split='trainval')
    train_dataset=c_d.crate()
    # with open('majomoji.pickle', mode='rb') as f:
    #     train_dataset=pickle.load(f)
    print('data_size = ',len(train_dataset))
    print('load')

    # データ拡張
    transformed_train_dataset = TransformDataset(train_dataset, Transform(model.coder, model.insize, model.mean))

    # イテレーターの設定
    train_iter = chainer.iterators.MultiprocessIterator(transformed_train_dataset, batchsize)

    test = VOCBboxDataset(
        year='2007', split='test',
        use_difficult=True, return_difficult=True)
    test_iter = chainer.iterators.SerialIterator(
        test, batchsize, repeat=False, shuffle=False)

    # オプティマイザーの設定
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    # アップデーターの設定
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # トレーナーの設定
    trainer = training.Trainer(updater, (epoch_max, 'epoch'), out_num)
    trainer.extend(extensions.ExponentialShift('lr', lr_decay_rate, init=initial_lr), trigger=triggers.ManualScheduleTrigger(lr_decay_timing, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'lr', 'main/loss', 'main/loss/loc', 'main/loss/conf', 'validation/main/map', 'elapsed_time']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=5))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'main/loss/loc', 'main/loss/conf'],
                'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['validation/main/map'],
                'epoch', file_name='accuracy.png'))

    # 学習実行
    print('start')
    trainer.run()

    # 学習データの保存
    model.to_cpu()
    serializers.save_npz('33my_ssd_model.npz', model)


if __name__ == '__main__':
    main()