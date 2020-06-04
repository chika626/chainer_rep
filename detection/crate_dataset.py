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

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation


cut_size = 512

def contrast(imgs,step):
    num=[]
    for i in range(step-1):
        num.append(pow(2,(i+1)))
        num.append(pow(1/2,(i+1)))
    images = []
    for img in imgs:
        for n in num:
            con_img=ImageEnhance.Contrast(img).enhance(n)
            images.append(con_img)
    return images

def saturation(imgs,step):
    num=[]
    for i in range(step-1):
        num.append(pow(2,(i+1)))
        num.append(pow(1/2,(i+1)))
    images = []
    for img in imgs:
        for n in num:
            con_img=ImageEnhance.Color(img).enhance(n)
            images.append(con_img)
    return images

def brightness(imgs,step):
    num=[]
    for i in range(step-1):
        num.append(pow(2,(i+1)))
        num.append(pow(1/2,(i+1)))
    images = []
    for img in imgs:
        for n in num:
            con_img=ImageEnhance.Brightness(img).enhance(n)
            images.append(con_img)
    return images


def one_take_json(json_path):
    with open(json_path) as f:
        result=json.load(f)

    img_path='majomoji/Image/'+result['asset']['name']
    img=Image.open(img_path).convert('RGB')

    # 1つのjsonにいくつも領域があるのでforで回す
    M=len(result['regions'])
    res_imgs,res_bboxes,res_labels=[],[],[]
    for i in range(M):
        # 4座標の加工 intなので切り捨てと切り上げで大きく取ることで対応
        x0=math.floor(result['regions'][i]['points'][0]['x'])
        y0=math.floor(result['regions'][i]['points'][0]['y'])
        x1=math.ceil(result['regions'][i]['points'][2]['x'])
        y1=math.ceil(result['regions'][i]['points'][2]['y'])
        # 中心とって正方形で対応
        dy=y1-y0
        dx=x1-x0
        dxy=max(dy,dx) / 2
        center_y=(y1+y0)/2
        center_x=(x1+x0)/2
        bbox=[0,0,dxy*2,dxy*2 ]
        
        take_img=img.crop((center_x - dxy , center_y - dxy , center_x + dxy , center_y + dxy))
        res_imgs.append(take_img)
        bboxes=[bbox]
        res_bboxes.append(bboxes)
        c=result['regions'][i]['tags'][0]
        c=ord(c)
        labels=[c-65]
        res_labels.append(labels)
    res_bboxes=np.array(res_bboxes,'f')
    res_labels=np.array(res_labels,'i')
    return res_imgs,res_bboxes,res_labels

def cut_512(json_path):
    # json開く
    with open(json_path) as f:
        result=json.load(f)

    # 画像をとる
    img_path='majomoji/Image/'+result['asset']['name']
    img=Image.open(img_path).convert('RGB')
    # img=Image.open(img_path).convert("L")
    # img=img.convert("RGB")

    # 画像サイズとっておく
    H , W = result['asset']['size']['height'] , result['asset']['size']['width']

    # bboxesを先に全部回収しておく
    M=len(result['regions'])
    bboxes=[]
    res_center_label=[]

    # 使用済みのものは再度選出しない
    used_flag = []
    for i in range(M):
        # 4座標とる
        x0=math.floor(result['regions'][i]['points'][0]['x'])
        y0=math.floor(result['regions'][i]['points'][0]['y'])
        x1=math.ceil(result['regions'][i]['points'][2]['x'])
        y1=math.ceil(result['regions'][i]['points'][2]['y'])
        bbox=[y0,x0,y1,x1]
        c=result['regions'][i]['tags'][0]
        c=ord(c)
        used_flag.append(False)
        res_center_label.append(c-65)
        bboxes.append(bbox)

    # bboxの全部を列挙できたので
    # 各bboxの中心から512*512の画像にする
    res_imgs,res_bboxes,res_labels=[],[],[]
    for i in range(M):
        # いちど選出されたものは見ないようにする
        if used_flag[i] == True:
           continue
        bbox=bboxes[i]
        # 中心とる
        center_y=(bbox[2]+bbox[0])/2
        center_x=(bbox[3]+bbox[1])/2
        # ここから256ずつの領域を切り出す
        d_xy=cut_size/2
        size_xy=[center_x-d_xy,center_y-d_xy,center_x+d_xy,center_y+d_xy]
        img_512=img.crop((size_xy[0],size_xy[1],size_xy[2],size_xy[3]))
        res_imgs.append(img_512)
        # この画像の中に含まれる文字を全部摘出する
        bboxes_512 , labels_512=[],[]
        for j in range(M):
            # 使用済みは見ないようにする
            if size_xy[0] < bboxes[j][1] and size_xy[1] < bboxes[j][0] and size_xy[2] > bboxes[j][3] and size_xy[3] > bboxes[j][2]:
                # 内包されるbboxなので追加する
                d_bbox=[bboxes[j][0]-size_xy[1],bboxes[j][1]-size_xy[0],bboxes[j][2]-size_xy[1],bboxes[j][3]-size_xy[0]]
                bboxes_512.append(d_bbox)
                c=result['regions'][j]['tags'][0]
                c=ord(c)
                labels_512.append(c-65)
                used_flag[j] = True
        bboxes_512=np.array(bboxes_512,'f')
        labels_512=np.array(labels_512,'i')
        res_bboxes.append(bboxes_512)
        res_labels.append(labels_512)
    
    return res_imgs,res_bboxes,res_labels

def crate(function,inf_con=False,inf_sat=False,inf_bri=False):
     # json一覧を取得
    json_path=glob.glob("majomoji/json/*")
    majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    dictionary=[0]*26
    imgs, bboxes, labels = [], [], []
    # 全jsonについてやってく
    for s in json_path:
        # 512ごとに切り出す
        imgs_512,bboxes_512,labels_512=function(s)
        for i in range(len(imgs_512)):
            # 512切り出し画像を1枚とる
            img=imgs_512[i]
            for j in labels_512[i]:
                dictionary[j]+=1

            # 手加える
            change_imgs = [img]
            t_img=[img]
            if inf_con:
                change_imgs.extend(contrast(t_img,2))
            if inf_sat:
                change_imgs.extend(saturation(t_img,2))
            if inf_bri:
                change_imgs.extend(brightness(t_img,2))

            images_array=[]
            for t_image in change_imgs:
                tra_img=np.transpose(t_image,(2,0,1))
                tra_img=np.array(tra_img,'f')
                images_array.append(tra_img)

            for image in images_array:
                imgs.append(image)
                bboxes.append(bboxes_512[i])
                labels.append(labels_512[i])
    # datasetに変換
    dataset=chainer.datasets.TupleDataset(imgs,bboxes,labels)

    print('crate!')

    for i in range(26):
        if i % 10 == 9:
            print('[',majomoji_label[i],':',str(dictionary[i]).rjust(3),']')
        else:
            print('[',majomoji_label[i],':',str(dictionary[i]).rjust(3),']  ',end='')
    print()

    return dataset



def main():
    # ここ呼んだら全画像を作成して保存させる
    # json一覧を取得
    json_path=glob.glob("majomoji/json/*")
    majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    dictionary=[0]*26
    imgs, bboxes, labels = [], [], []

    # 選択
    take_function=cut_512
    inf_con = False
    inf_sat = False
    inf_bri = False

    counter = 0
    # 全jsonについてやってく
    for s in json_path:
        # 切り出し手法によって「画像」「bbox群」「label群」「その画像に含まれるlabelの種類と数」を返させる
        imgs_512,bboxes_512,labels_512=take_function(s)
        for i in range(len(imgs_512)):
            # 512切り出し画像を1枚とる
            img=imgs_512[i]

            # 2値化
            img=img.convert('L')
            img=np.asarray(img)
            ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            img=Image.fromarray(img)
            img=img.convert('RGB')

            # 実際に合ってるか確認したい
            cambass = ImageDraw.Draw(img)
            for j in range(len(bboxes_512[i])):
                # j文字あるって意味
                y0 = bboxes_512[i][j][0]
                x0 = bboxes_512[i][j][1]
                y1 = bboxes_512[i][j][2]
                x1 = bboxes_512[i][j][3]
                # cambass.line( ( (x0,y0) , (x0,y1) , (x1,y1) ,(x1,y0) , (x0 , y0) ) , fill=(255, 255, 0) ,width=10)
            img.save('crop/data_{}.png'.format(counter))
            counter+=1
            # 画像に含まれる全部の文字を学習数としてカウントする
            for j in labels_512[i]:
                dictionary[j]+=1

            # 手加える
            change_imgs = [img]
            t_img=[img]
            if inf_con:
                change_imgs.extend(contrast(t_img,2))
            if inf_sat:
                change_imgs.extend(saturation(t_img,2))
            if inf_bri:
                change_imgs.extend(brightness(t_img,2))

            images_array=[]
            for t_image in change_imgs:
                tra_img=np.transpose(t_image,(2,0,1))
                tra_img=np.array(tra_img,'f')
                images_array.append(tra_img)

            for image in images_array:
                imgs.append(image)
                bboxes.append(bboxes_512[i])
                labels.append(labels_512[i])

    # datasetに変換
    dataset=chainer.datasets.TupleDataset(imgs,bboxes,labels)

    print('crate!')

    # 画像のうちわけ、含まれる数まで見たほうがいいと思うけどなぁ
    for i in range(26):
        if i % 10 == 9:
            print(majomoji_label[i],':',str(dictionary[i]).rjust(3))
        else:
            print(majomoji_label[i],':',str(dictionary[i]).rjust(3),'  ',end='')
    print()

if __name__ == '__main__':
    main()