import json
import math
from PIL import Image,ImageDraw
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
from chainercv.utils import read_image

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation
import queue

def run(img):
    # c , H , W = img.shape
    H,W = img.size
    img = np.asarray(img)

    # 変換後データ配列
    transed = Image.new('RGB',(H,W))
    for x in range(H):
        for y in range(W):
            transed.putpixel((x,y),(255,255,255))

    for x in range(H):
        for y in range(W):
            if x + 1 == H or y + 1 == W:
                break

            if img[y][x][0] != img[y][x+1][0]:
                transed.putpixel((x,y),(0,0,0))
    for y in range(W):
        for x in range(H):
            if x + 1 == H or y + 1 == W:
                break

            if img[y][x][0] != img[y+1][x][0]:
                transed.putpixel((x,y),(0,0,0))            

    return transed

    

def main():
    # # 単一の場合のコード
    # img = Image.open('cont/transed/X.jpg')
    # img=img.convert('L')
    # img=np.asarray(img)
    # ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # img=Image.fromarray(img)
    # img=img.convert('RGB')
    # transed = run(img)
    # transed.save('transec_0.png')
    # return

    # 大量変換機
    img_path=glob.glob("cont/crop/*")
    counter=0
    for path in img_path:
        img = Image.open(path)
        transed = run(img)
        transed.save('transec_{}.png'.format(counter))
        counter+=1



if __name__ == '__main__':
    main()