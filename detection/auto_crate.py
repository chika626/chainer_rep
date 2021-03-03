import glob
import random
import time
import chainer
import numpy as np
from PIL import Image


# imgs , bboxes ,labels がデータセットの中身なので毎回作るくらいならデータ上で簡潔させる
folder_path='fill_moji'
input_size = 300



# 全データを取得する
def input():
    alp_data = []
    alp_size = []
    for x in range(ord('A'),ord('Z')+1):
        path = folder_path+'/'+ chr(x) +'/*'
        all_img_path=glob.glob(path)
        alp_data.append(all_img_path)
        alp_size.append(len(all_img_path))
    return alp_data , alp_size

# 画像 下地色 文字色 で着色を行う
def paint(img,back_color,moji_color):
    H,W = img.size
    img = np.asarray(img)
    painted = Image.new('RGB',(H,W))
    for x in range(H):
        for y in range(W):
            painted.putpixel((x,y),(back_color[0],back_color[1],back_color[2]))
    for y in range(H):
        for x in range(W):
            if img[x][y][0] == 0:
                painted.putpixel((y,x),(moji_color[0],moji_color[1],moji_color[2]))
    return painted

# size x size の bc画像 に painted を合成します
def transe(painted,back_color,size):
    H,W = painted.size
    bc = (back_color[0],back_color[1],back_color[2])
    transed = Image.new('RGB',(size,size),bc)
    transed.paste(painted,(0,0))

    # painted = np.asarray(painted)
    # for y in range(size):
    #     for x in range(size):
    #         transed.putpixel((y,x),(back_color[0],back_color[1],back_color[2]))
    # for y in range(H):
    #     for x in range(W):
    #         transed.putpixel((y,x),(painted[x][y][0],painted[x][y][1],painted[x][y][2]))
    return transed

# bbox を盗る
def take_bbox(transed,bc):

    # start = time.time()

    H,W = transed.size
    y0,x0,y1,x1 = 0,0,0,0
    transed = np.asarray(transed)

    # asa = time .time()
    # print('load:{0}'.format(asa- start))

    # 上側
    for y in range(H):
        ok_flag = True
        for x in range(W):
            if transed[y][x][0] != bc[0] or transed[y][x][1] != bc[1] or transed[y][x][2] != bc[2]:
                ok_flag = False
                break
        if ok_flag:
            y0 = y
        else:
            break
    # 下側
    for y in range(H):
        ok_flag = True
        for x in range(W):
            if transed[(H-1-y)][x][0] != bc[0] or transed[(H-1-y)][x][1] != bc[1] or transed[(H-1-y)][x][2] != bc[2]:
                ok_flag = False
                break
        if ok_flag:
            y1 = (H-1-y)
        else:
            break
    # 左側
    for x in range(W):
        ok_flag = True
        for y in range(H):
            if transed[y][x][0] != bc[0] or transed[y][x][1] != bc[1] or transed[y][x][2] != bc[2]:
                ok_flag = False
                break
        if ok_flag:
            x0 = x
        else:
            break
    # 右側
    for x in range(W):
        ok_flag = True
        for y in range(H):
            if transed[y][(W - 1 -x)][0] != bc[0] or transed[y][(W - 1 -x)][1] != bc[1] or transed[y][(W - 1 -x)][2] != bc[2]:
                ok_flag = False
                break
        if ok_flag:
            x1 = (W - 1 -x)
        else:
            break

    # ser = time.time()
    # print('serach:{0}'.format(ser-asa))

    bbox = [y0,x0,y1,x1]
    bboxes = [bbox]
    bboxes = np.array(bboxes,'f')

    # end = time.time()
    # print('end:{0}'.format(end - ser))

    return bboxes

def take_bbox_b(transed,bc):
    H,W = transed.size
    bbox = [0,0,H,W]
    bboxes = [bbox]
    bboxes = np.array(bboxes,'f')
    return bboxes

def take_label(x):
    x = ord(x)-65
    labels = [x]
    labels=np.array(labels,'i')
    return labels



# max_data_size = 作成するデータ量
def crate(max_data_size):
    # 基本指針、塗り終わり画像から全画像ピック
    alphabet = [chr(c) for c in range(ord('A'),ord('Z')+1)]
    alp_data , alp_size = input()
    crate_data_size = 0

    # 拡大率の変更
    # 512 - 320 - 192 - 64 ???
    # size = [512,320,256,192,64]
    size = [256,192,64,32,16]

    imgs, bboxes, labels = [], [], []

    while crate_data_size < max_data_size:
        # A-Zの全てを1つずつ作る
        # まず背景色、文字色、ノイズ、サイズなどをキメる
        background_color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        moji_color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        while(background_color == moji_color):
            moji_color=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        
        # 全alphabetを加工する
        for x in alphabet:
            i = ord(x) - ord('A')
            j = crate_data_size % alp_size[i]
            img = Image.open(alp_data[i][j]).convert('RGB')

            # 着色
            painted = paint(img,background_color,moji_color)
            
            # resize
            painted = painted.resize((size[crate_data_size % 5],size[crate_data_size % 5]))

            # 拡大処理
            transed = transe(painted,background_color,input_size)

            # bbox をとる
            one_bboxes = take_bbox_b(transed,background_color)

            # label 形式合わせる
            one_labels = take_label(x)

            bboxes.append(one_bboxes)
            labels.append(one_labels)
            img = np.transpose(img,(2,0,1))
            img = np.array(img,'f')
            imgs.append(img)

        crate_data_size += 1

    dataset=chainer.datasets.TupleDataset(imgs,bboxes,labels)
    print('auto crate!!!')
    return dataset
    

def main():
    dt = crate(30)

if __name__ == '__main__':
    main()