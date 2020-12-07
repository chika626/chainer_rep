import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import math
import glob
import os
import datetime
import numpy as np
from chainer import serializers
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD512
from chainercv.links import SSD300
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox
import cv2
import cv2 as cv

# model_name = 'model/2020_6_2.npz'
# model_name = 'model/2020_7_2.npz'
model_name = 'model/2020_9_18.npz'
result_path = "majomoji/inference/result"
color = [255.0, .0, .0]

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

# ひとつの画像に対して推論をする場所
def take_image(img_path):
    # 画像をとる
    img = read_image(img_path)

    # 推論実行
    majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    model = SSD512(n_fg_class=len(majomoji_label))
    serializers.load_npz(model_name,model)
    bboxes, labels, scores = run(img,model)

    vis_bbox(img, bboxes, labels, scores,
        label_names=majomoji_label)
    plt.show()

    return bboxes,labels,scores

# 疑似Main文
def start_inference():
    # まず今回の結果を保存するためのフォルダを生成する
    dt_now=datetime.datetime.now()
    folder_name=result_path+'/{}_{}_{}'.format(dt_now.year,dt_now.month,dt_now.day)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # フォルダ内の全てに対して推論を行う
    all_img_path=glob.glob("majomoji/inference/Image/*")
    for img_path in all_img_path:
        print('inference ... [' , img_path , ']')
        bboxes,labels,scores = take_image(img_path)
        # txtに保持
        img_name = img_path.split('\\')[-1].split('.')[0]
        print(img_name)
        path_and_name = folder_name+'/'+img_name+'.txt'
        with open(path_and_name,mode='w') as f:
            f.write('{}\n'.format(bboxes))
            f.write('{}\n'.format(labels))
            f.write('{}\n'.format(scores))


    return 0

def dwar_frame(bboxes,labels,img):
    
    for box in bboxes:
        # 1box = 4座標
        x0 = int(box[0])
        x1 = int(box[2])
        y0 = int(box[1])
        y1 = int(box[3])
        # 横
        for x in range(int(x1)-int(x0)):
            for i in range(3):
                img[i][x0+x][y0] = color[i]
                img[i][x0+x][y1] = color[i]
        # 縦
        for y in range(y1-y0):
            for i in range(3):
                img[i][x0][y0+y] = color[i]
                img[i][x1][y0+y] = color[i]
    
    # opencvで扱えるように変換
    img = trans_img_cv2(img)
    majomoji_label=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    for i in range(len(labels)):
        # 文字書き込み
        cv.putText(img, majomoji_label[labels[i]], 
        (bboxes[i][1], bboxes[i][2]), 
        cv.FONT_HERSHEY_PLAIN, 5, color, 5, cv.LINE_AA)

    return img

def trans_img_cv2(img):
    buf = np.asanyarray(img, dtype=np.uint8).transpose(1, 2, 0)
    dst = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    return dst

def discord_inf(png):
    img = read_image(png)

    # 学習済みmodelを渡す
    majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    model = SSD512(n_fg_class=len(majomoji_label))

    # model のロード
    serializers.load_npz(model_name,model)

    # 推論の実行
    bboxes, labels, scores = run(img,model)

    print("推論終了")

    # 加工も行って画像を返す
    # [(RGB),(y),(x)]
    # 線入れ関数
    
    d_img = dwar_frame(bboxes,labels,img)

    cv2.imwrite("fin_inf.jpg",d_img)


def main():

    discord_inf("test016.PNG")

    # start_inference()

    # 推論させたい画像の選択
    # img = read_image('majomoji/Image/test016.PNG')

    # 学習済みmodelを渡す
    # majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    # model = SSD512(n_fg_class=len(majomoji_label))

    # model のロード
    # serializers.load_npz('model/2020_3_30_con.npz',model)
    # serializers.load_npz('model/2020_5_27.npz',model)

    # 推論の実行
    # bboxes, labels, scores = run(img,model)
    # vis_bbox(img, bboxes, labels, scores,
    #     label_names=majomoji_label)
    # plt.show()


if __name__ == '__main__':
    main()