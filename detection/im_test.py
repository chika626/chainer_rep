import json
import math
from PIL import Image
import pandas as pd
import glob
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
from PIL import ImageEnhance

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer import datasets
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


majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"

json_path='majomoji/json/f0abfcf5ff4fb229795dc9c5ec5b39f0-asset.json'

img_path='majomoji/Image/Image001.png'

img=Image.open(img_path).convert('L')
img.show()

# 2値化
img=np.asarray(img)

ret2, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("otsu", img_otsu)
cv2.waitKey()
cv2.destroyAllWindows()

img=Image.fromarray(img_otsu)


img.show()