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
    infer_net = ResNet()

    # ここ変えて推論
    model_name = ''
    serializers.load_npz(model_name, infer_net)


    ini = configparser.ConfigParser()  
    ini.read('./config.ini', 'UTF-8')
    gpu_id = int(ini['inference']['gpu_id'])

    if gpu_id >= 0:
        infer_net.to_gpu(gpu_id)

    # 1つ目のテストデータを取り出します
    x, t = test[0]  #  tは使わない

    # どんな画像か表示してみます
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()

    # ミニバッチの形にする（複数の画像をまとめて推論に使いたい場合は、サイズnのミニバッチにしてまとめればよい）
    print('元の形：', x.shape, end=' -> ')

    x = x[None, ...]

    print('ミニバッチの形にしたあと：', x.shape)

    # ネットワークと同じデバイス上にデータを送る
    x = infer_net.xp.asarray(x)

    # モデルのforward関数に渡す
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = infer_net(x)

    # Variable形式で出てくるので中身を取り出す
    y = y.array

    # 結果をCPUに送る
    y = to_cpu(y)

    # 予測確率の最大値のインデックスを見る
    pred_label = y.argmax(axis=1)

    print('ネットワークの予測:', pred_label[0])