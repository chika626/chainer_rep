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
import configparser

class CIFAR10Augmented(chainer.dataset.DatasetMixin):

    def __init__(self, split='train', train_ratio=0.9):
        train_val, test_data = cifar.get_cifar10()
        train_size = int(len(train_val) * train_ratio)
        train_data, valid_data = chainer.datasets.split_dataset_random(train_val, train_size, seed=0)
        if split == 'train':
            self.data = train_data
        elif split == 'valid':
            self.data = valid_data
        elif split == 'test':
            self.data = test_data
        else:
            raise ValueError("'split' argument should be either 'train', 'valid', or 'test'. But {} was given.".format(split))

        self.split = split
        self.random_crop = 4

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        x, t = self.data[i]
        if self.split == 'train':
            x = x.transpose(1, 2, 0)
            h, w, _ = x.shape
            x_offset = np.random.randint(self.random_crop)
            y_offset = np.random.randint(self.random_crop)
            x = x[y_offset:y_offset + h - self.random_crop,
                  x_offset:x_offset + w - self.random_crop]
            if np.random.rand() > 0.5:
                x = np.fliplr(x)
            x = x.transpose(2, 0, 1)

        return x, t