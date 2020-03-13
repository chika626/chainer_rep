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

def take_json(json_path):
    # jsonのload
    with open(json_path) as f:
        result=json.load(f)
    
    # 画像とる
    img_path='majomoji/Image/'+result['asset']['name']
    img=Image.open(img_path).resize((result['asset']['size']['width'],result['asset']['size']['height'])).convert('RGB')
    img=np.transpose(img,(2,0,1))
    
    # 1つのjsonにいくつも領域があるのでforで回す
    M=len(result['regions'])
    bboxes ,labels = [] ,[]
    for i in range(M):
        # 4座標の加工 intなので切り捨てと切り上げで大きく取ることで対応
        x0=math.floor(result['regions'][i]['points'][0]['x'])
        y0=math.floor(result['regions'][i]['points'][0]['y'])
        x1=math.ceil(result['regions'][i]['points'][2]['x'])
        y1=math.ceil(result['regions'][i]['points'][2]['y'])
        bbox=[y0,x0,y1,x1]
        bboxes.append(bbox)
        # Labelもとる
        c=result['regions'][i]['tags'][0]
        c=ord(c)
        labels.append(c-65)

    # 数列に変換
    img=np.array(img,'f')
    bboxes=np.array(bboxes,'f')
    labels=np.array(labels,'i')

    return img_path,img,bboxes,labels

def contrast(img):
    min_contrast=0.1
    max_contrast=2.0
    d_contrast=0.3

    num = min_contrast
    images = []
    while num < max_contrast:
        con_img=ImageEnhance.Contrast(img).enhance(num)
        con_img=np.transpose(con_img,(2,0,1))
        con_img=np.array(con_img,'f')
        images.append(con_img)
        num += d_contrast

    return images

def saturation(img):
    min_saturation=0.1
    max_saturation=2.0
    d_saturation=0.3

    num = min_saturation
    images = []
    while num < max_saturation:
        con_img=ImageEnhance.Color(img).enhance(num)
        con_img=np.transpose(con_img,(2,0,1))
        con_img=np.array(con_img,'f')
        images.append(con_img)
        num += d_saturation

    return images

def brightness(img):
    min_brightness=0.1
    max_brightness=2.0
    d_brightness=0.3

    num = min_brightness
    images = []
    while num < max_brightness:
        con_img=ImageEnhance.Brightness(img).enhance(num)
        con_img=np.transpose(con_img,(2,0,1))
        con_img=np.array(con_img,'f')
        images.append(con_img)
        num += d_brightness

    return images



def take_path_box_label(json_path):
    # jsonのload
    with open(json_path) as f:
        result=json.load(f)
    
    # 画像とる
    img_path='majomoji/Image/'+result['asset']['name']
    
    # 1つのjsonにいくつも領域があるのでforで回す
    M=len(result['regions'])
    bboxes ,labels = [] ,[]
    for i in range(M):
        # 4座標の加工 intなので切り捨てと切り上げで大きく取ることで対応
        x0=math.floor(result['regions'][i]['points'][0]['x'])
        y0=math.floor(result['regions'][i]['points'][0]['y'])
        x1=math.ceil(result['regions'][i]['points'][2]['x'])
        y1=math.ceil(result['regions'][i]['points'][2]['y'])
        bbox=[y0,x0,y1,x1]
        bboxes.append(bbox)
        # Labelもとる
        c=result['regions'][i]['tags'][0]
        c=ord(c)
        labels.append(c-65)

    # 数列に変換
    bboxes=np.array(bboxes,'f')
    labels=np.array(labels,'i')

    return img_path,bboxes,labels

def crate():
    # json一覧を取得
    json_path=glob.glob("majomoji/json/*")

    imgs, bboxes, labels = [], [], []
    # 全jsonについてやってく
    for s in json_path:
        # path , bbox , label を取る
        path,bbox,label=take_path_box_label(s)

        # 変換元画像を取っておく
        img=Image.open(path).convert('RGB')
        for image in contrast(img):
            imgs.append(image)
            bboxes.append(bbox)
            labels.append(label)
        for image in saturation(img):
            imgs.append(image)
            bboxes.append(bbox)
            labels.append(label)
        for image in brightness(img):
            imgs.append(image)
            bboxes.append(bbox)
            labels.append(label)
 
    # datasetに変換
    dataset=chainer.datasets.TupleDataset(imgs,bboxes,labels)

    print('crate!')

    return dataset

def _crate():
    # json一覧を取得
    json_path=glob.glob("majomoji/json/*")

    imgs, bboxes, labels = [], [], []
    # 全jsonについてやってく
    for s in json_path:
        # path , bbox , label を取る
        path,bbox,label=take_path_box_label(s)

        # 変換元画像を取っておく
        img=Image.open(path).convert('RGB')
        img=np.transpose(img,(2,0,1))
        img=np.array(img,'f')

        imgs.append(img)
        bboxes.append(bbox)
        labels.append(label)

 
    # datasetに変換
    dataset=chainer.datasets.TupleDataset(imgs,bboxes,labels)

    print('crate!')

    return dataset




def main():
    # json一覧を取得
    json_path=glob.glob("majomoji/json/*")

    imgs, bboxes, labels = [], [], []
    # 全jsonについてやってく
    for s in json_path:
        img_path,a,b,c = take_json(s)
        for img in contrast(img_path):
            imgs.append(img)
            bboxes.append(b)
            labels.append(c)
        for img in saturation(img_path):
            imgs.append(img)
            bboxes.append(b)
            labels.append(c)
        for img in brightness(img_path):
            imgs.append(img)
            bboxes.append(b)
            labels.append(c)

    # datasetに変換
    dataset=chainer.datasets.TupleDataset(imgs,bboxes,labels)

    print(len(dataset))

    with open('majomoji.pickle', mode='wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()