import chainer
from chainer.datasets import mnist
import chainer.links as L
import math
import cupy
import chainer.cuda
import chainer.functions as F
from chainer import iterators
#from chainercv.transforms import resize
from chainer.datasets import TransformDataset
from chainer import optimizers

class ConvBlock(chainer.Chain):

    def __init__(self, n_ch, pool_drop=False):
        w = chainer.initializers.HeNormal()
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, n_ch, 3, 1, 1, nobias=True, initialW=w)
            self.bn = L.BatchNormalization(n_ch)
        self.pool_drop = pool_drop

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        if self.pool_drop:
            h = F.max_pooling_2d(h, 2, 2)
            h = F.dropout(h, ratio=0.25)
        return h

class LinearBlock(chainer.Chain):

    def __init__(self, drop=False):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, 1024, initialW=w)
        self.drop = drop

    def __call__(self, x):
        h = F.relu(self.fc(x))
        if self.drop:
            h = F.dropout(h)
        return h
        
class DeepCNN(chainer.ChainList):

    def __init__(self, n_output):
        super(DeepCNN, self).__init__(
            ConvBlock(64),
            ConvBlock(64, True),
            ConvBlock(128),
            ConvBlock(128, True),
            ConvBlock(256),
            ConvBlock(256),
            ConvBlock(256),
            ConvBlock(256, True),
            LinearBlock(),
            LinearBlock(),
            L.Linear(None, n_output)
        )

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x

# ここからResNet
# 他人実装ResNet、とりあえず回しながら改変

# class ResBlock(chainer.Chain):
#     def __init__(self, n_in, n_out, stride=1, ksize=1):
#         w = math.sqrt(2)
#         super(ResBlock, self).__init__(
#             conv1=L.Convolution2D(n_in, n_out, 3, stride, 1, w),
#             bn1=L.BatchNormalization(n_out),
#             conv2=L.Convolution2D(n_out, n_out, 3, 1, 1, w),
#             bn2=L.BatchNormalization(n_out),
#         )
#     def __call__(self, x, train):
#         h = F.relu(self.bn1(self.conv1(x), test=not train))
#         h = self.bn2(self.conv2(h), test=not train)
#         if x.data.shape != h.data.shape:
#             xp = cupy.get_array_module(x.data)
#             n, c, hh, ww = x.data.shape
#             pad_c = h.data.shape[1] - c
#             p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
#             p = chainer.Variable(p, volatile=not train)
#             x = F.concat((p, x))
#             if x.data.shape[2:] != h.data.shape[2:]:
#                 x = F.average_pooling_2d(x, 1, 2)
#         return F.relu(h + x)

# class ResNet(chainer.Chain):
#     def __init__(self, block_class, n=18):
#         super(ResNet, self).__init__()
#         w = math.sqrt(2)
#         links = [('conv1', L.Convolution2D(3, 16, 3, 1, 0, w))]
#         links += [('bn1', L.BatchNormalization(16))]
#         for i in range(n):
#             links += [('res{}'.format(len(links)), block_class(16, 16))]
#         for i in range(n):
#             links += [('res{}'.format(len(links)),
#                        block_class(32 if i > 0 else 16, 32,
#                                    1 if i > 0 else 2))]
#         for i in range(n):
#             links += [('res{}'.format(len(links)),
#                        block_class(64 if i > 0 else 32, 64,
#                                    1 if i > 0 else 2))]
#         links += [('_apool{}'.format(len(links)),
#                    F.AveragePooling2D(6, 1, 0, False, True))]
#         links += [('fc{}'.format(len(links)),
#                    L.Linear(64, 10))]
#         for link in links:
#             if not link[0].startswith('_'):
#                 self.add_link(*link)
#         self.forward = links
#         self.train = True
#     def __call__(self, x, t):
#         for name, f in self.forward:
#             if 'res' in name:
#                 x = f(x, self.train)
#             else:
#                 x = f(x)
#         if self.train:
#             self.loss = F.softmax_cross_entropy(x, t)
#             self.accuracy = F.accuracy(x, t)
#             return self.loss
#         else:
#             return x

        

# # 独自ResNet

# class ResBlock(chainer.Chain):
#     def __init__(self, n_in, n_out, stride=1, ksize=1):
#         w = math.sqrt(2)
#         super(ResBlock, self).__init__(
#             conv1=L.Convolution2D(n_in, n_out, 3, stride, 1, w),
#             bn1=L.BatchNormalization(n_out),
#             conv2=L.Convolution2D(n_out, n_out, 3, 1, 1, w),
#             bn2=L.BatchNormalization(n_out),
#         )
#     def __call__(self, x, train):
#         h = F.relu(self.bn1(self.conv1(x), test=not train))
#         h = self.bn2(self.conv2(h), test=not train)
#         if x.data.shape != h.data.shape:
#             xp = cupy.get_array_module(x.data)
#             n, c, hh, ww = x.data.shape
#             pad_c = h.data.shape[1] - c
#             p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
#             p = chainer.Variable(p, volatile=not train)
#             x = F.concat((p, x))
#             if x.data.shape[2:] != h.data.shape[2:]:
#                 x = F.average_pooling_2d(x, 1, 2)
#         return F.relu(h + x)

# class ResBox(chainer.Chain):
#     def __init__(self,n_out,n_last,stride=1,ksize=1):
#         w = chainer.initializers.HeNormal()
#         super(ResBox,self).__init__(
#                 # ksize,stride,pad
#                 # 畳み込み層 
#                 # (1x1 conv equivalent)
#                 conv1=L.Convolution2D(None,n_out, 1, 1, 0,initialW=w),
#                 # (3x3 conv equivalent)
#                 conv2=L.Convolution2D(n_out,n_out, 3, 1 ,1,initialW=w),
#                 # (1x1 conv expansion)
#                 conv2=L.Convolution2D(n_out,n_last, 1, 1 ,1,initialW=w),
#                 # バッチ正規化
#                 bn1=L.BatchNormalization(n_out),
#                 bn2=L.BatchNormalization(n_out),
#                 bn3=L.BatchNormalization(n_out)
#         )
#     def __call__(self,x):
#         h=F.relu(self.bn1(self.conv1(x)))
#         h=

#          h = F.relu(self.bn(self.conv(x)))
#         if self.pool_drop:
#             h = F.max_pooling_2d(h, 2, 2)
#             h = F.dropout(h, ratio=0.25)