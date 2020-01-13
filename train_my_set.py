import chainer
import argparse
from chainer import training
from chainer.datasets import cifar
from chainer.datasets import mnist
import chainer.links as L
import chainer.functions as F
from chainer import iterators
import numpy as np
#from chainercv.transforms import resize
from chainer.datasets import TransformDataset
from chainer import optimizers
from chainer.datasets import LabeledImageDataset
from chainer.training import extensions
import train_network as Net
import data_set as dataset
import configparser
from chainer import serializers
import datetime



def train(network_object, batchsize=128, gpu_id=0, max_epoch=30, train_dataset=None, valid_dataset=None, test_dataset=None, postfix='', base_lr=0.01, lr_decay=None,out_dump=True):

    # 1. Dataset
    if train_dataset is None and valid_dataset is None and test_dataset is None:
        train_val, test = cifar.get_cifar10()
        train_size = int(len(train_val) * 0.9)
        train, valid = chainer.datasets.split_dataset_random(train_val, train_size, seed=0)
    else:
        train, valid, test = train_dataset, valid_dataset, test_dataset

#     # train = LabeledImageDataset('data/train/train_labels.txt', 'data/train/images')
#     # valid = LabeledImageDataset('data/train/valid_labels.txt', 'data/valid/images')
#     # #独自サイズに変更
#     # def transform(in_data):
#     #     img, label = in_data
#     #     img = resize(img, (224, 224))
#     #     return img, label
#     # #image labelの順で入ってる
#     # train = TransformDataset(train, transform)
#     # valid = TransformDataset(valid, transform)

    # 2. Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    valid_iter = iterators.MultiprocessIterator(valid, batchsize, False, False)

    # 3. Model
    net = L.Classifier(network_object)

    # 4. Optimizer
    optimizer = optimizers.MomentumSGD(lr=base_lr).setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # 5. Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # 6. Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_cifar1002_{}result'.format(network_object.__class__.__name__, postfix))

    # 7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())
    # OutPut Network.dot
    if out_dump:
        trainer.extend(extensions.dump_graph('main/loss'))


    # output infomation
    print('gpu_id : ',gpu_id)
    print('batchsize : ',batchsize)


    if lr_decay is not None:
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_decay)
    trainer.run()
    del trainer

    # 8. Evaluation
    test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)
    test_evaluator = extensions.Evaluator(test_iter, net, device=gpu_id)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])

    return net


def main():
    # chainer.cuda.set_max_workspace_size(512*1024*1024)
    # #chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
    # chainer.config.autotune = True

    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'UTF-8')
    s_0 = 'train'
    _batchsize = int(ini[s_0]['batchsize'])
    _max_epoch = int(ini[s_0]['max_epoch'])
    _gpu_id = int(ini[s_0]['gpu_id'])
    _postfix = ini[s_0]['postfix']
    _base_lr = float(ini[s_0]['base_lr'])
    _lr_decay = int(ini[s_0]['lr_decay'])
    _out_dump = bool(ini[s_0]['out_dump'])

    # これで動いてた
    # ResNet(最終出力数)
    # model = train(Net.ResNet(10),
    #  batchsize=_batchsize,
    #  gpu_id=_gpu_id,
    #  max_epoch=_max_epoch, 
    #  train_dataset=dataset.CIFAR10Augmented(),
    #  valid_dataset=dataset.CIFAR10Augmented('valid'),
    #  test_dataset=dataset.CIFAR10Augmented('test'),
    #  postfix=_postfix, 
    #  base_lr=_base_lr, 
    #  lr_decay=(_lr_decay, 'epoch'),
    #  out_dump=_out_dump
    # )

    # model = train(Net.DeepCNN(10))
    model = train(Net.ResNet(10))
    
    #output model
    dt_now = datetime.datetime.now()
    serializers.save_npz('{}_{}_{}.model'.format(dt_now.year,dt_now.month,dt_now.day),model)

    
    

if __name__ == "__main__":
    main()