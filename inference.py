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
import matplotlib.pyplot as plt
import random


def main():
    # Set up a neural network
    model = L.Classifier(Net.DeepCNN(10))
    chainer.serializers.load_npz('out.model', model)

    train, test = cifar.get_cifar10()
    # ランダムなテストデータを選ぶ
    m=len(test)
    n=random.randrange(m)
    x, t = test[n]
    print('label:', t)

    x = x[None, ...]
    y = model.predictor(x)
    y = y.data

    print('predicted_label:', y.argmax(axis=1)[0])



if __name__ == "__main__":
    main()