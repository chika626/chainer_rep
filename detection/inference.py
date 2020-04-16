import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import math
import numpy as np
from chainer import serializers
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD512
from chainercv.links import SSD300
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox

# 推論実行するやつ
def run(img,model):
    bboxes, labels, scores = model.predict([img])
    # 整数値をとりたいのでここで成形する
    take_bboxes=[]
    for i in range(len(bboxes[0])):
        y0=math.floor(bboxes[0][i][0])
        x0=math.floor(bboxes[0][i][1])
        y1=math.ceil(bboxes[0][i][2])
        x1=math.ceil(bboxes[0][i][3])
        bbox=[y0,x0,y1,x1]
        take_bboxes.append(bbox)
    take_bboxes=np.array(take_bboxes,'f')
    return take_bboxes, labels[0], scores[0]


def cut_img_512(img):
    imgs=[]
    cut_size=512
    H , W= img.size

    My = math.ceil(H/cut_size)
    Mx = math.ceil(W/cut_size)

    # normal
    for y in range(My):
        for x in range(Mx):
            sy,sx = cut_size*y , cut_size*x
            part_img=img.crop(( sy , sx , sy + cut_size , sx + cut_size))
            imgs.append(part_img)
    return imgs

def main():
    # 推論させたい画像の選択
    img = read_image('majomoji/Image/test027.PNG')
    # img = Image.open('majomoji/Image/test000.PNG').convert('RGB')
    # 512*512に分解する
    #imgs=cut_img_512(img)

    # 学習済みmodelを渡す
    majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    model = SSD512(n_fg_class=len(majomoji_label))

    # model のロード
    serializers.load_npz('model/2020_3_30_con.npz',model)

    # 推論の実行
    # bboxes, labels, scores = model.predict([img])
    # vis_bbox(img, bboxes[0], labels[0], scores[0],
    #      label_names=majomoji_label)
    bboxes, labels, scores = run(img,model)
    vis_bbox(img, bboxes, labels, scores,
        label_names=majomoji_label)
    plt.show()


if __name__ == '__main__':
    main()