import numpy as np

import chainer
from chainer import serializers
from chainercv.links import SSD512
import chainercv.links as C
import chainercv
import onnx_chainer


# model_name = 'model/2020_9_18.npz'
# majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
# model = SSD512(n_fg_class=len(majomoji_label))
# serializers.load_npz(model_name,model)

model = C.VGG16(pretrained_model='imagenet')

# ネットワークに流し込む擬似的なデータを用意する
x = np.zeros((1, 3, 224, 224), dtype=np.float32)

# 推論モードにする
chainer.config.train = False

onnx_chainer.export(model, x, filename='vgg16.onnx')