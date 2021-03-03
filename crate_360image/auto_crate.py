import glob
import random
import time
import chainer
import numpy as np
from PIL import Image
import cv2
import math


# PIL型 -> OpenCV型
def pil2cv(image):
    
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    return new_image
# OpenCV型 -> PIL型
def cv2pil(image):
    
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    new_image = Image.fromarray(new_image)
    return new_image

def main():
    crate('IMG_3017.png','0.png')
    # for i in range(65, 65+26):
    #     image_path = chr(i)+'.png'
    #     image_path2 = str(i)+'.png'
    #     crate(image_path,image_path2)


def crate(image_path,image_path2):
    default_image = Image.open(image_path)
    w,h = default_image.size
    # print('w : ',w)
    # print('h : ',h)

    # 画像の下地を作る
    one_rect = math.ceil(w * math.sqrt(2))
    # print('rect : ',one_rect)
    new_img = img=np.zeros((one_rect*6, one_rect*6, 3), np.uint8)

    # 元画像を回転対応させる(PIL -> cv2)
    img = img=np.zeros((one_rect, one_rect, 3), np.uint8)
    img[:,:,0:3]=[255,255,255]
    img = pil2cv(img)
    def_img = pil2cv(default_image)
    xy = (one_rect-w)//2
    img[xy:xy+h,xy:xy+w] = def_img
    img = cv2pil(img)

    # 回転させながら埋め込んでいく
    for i in range(36):
        # 角度
        deg = i*10
        y = (i//6)*one_rect
        x = (i%6)*one_rect
        # 回転(cv2に変換)
        rot = img.rotate(deg,fillcolor=(255, 255, 255))
        rot = pil2cv(rot)
        # 書き込み
        new_img[y:y+one_rect , x:x+one_rect] = rot

    print('end : ',image_path)
    cv2.imwrite(image_path2,new_img)

if __name__ == '__main__':
    main()