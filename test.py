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

def main():
    print('Hello World!')

