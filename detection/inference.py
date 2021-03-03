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
from collections import deque
import time
import multiprocessing
import concurrent.futures as confu

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

# 並列推論
def multi_run(img,model,send_rev,rad,center):
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
    take_bboxes = moldingrotation(rad,center,take_bboxes)
    send_rev.send([take_bboxes, labels[0], scores[0]])
    # return take_bboxes, labels[0], scores[0]

def multi_run2(img,model,rad,center):
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
    take_bboxes = moldingrotation(rad,center,take_bboxes)
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

# 提案領域とか文字を書き込む関数
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
# PIL型 -> ChainerCV型
def pilccv(pilimg):
    img = np.asarray(pilimg, dtype=np.float32)
    # transpose (H, W, C) -> (C, H, W)
    return img.transpose((2, 0, 1))

def molrot(_rad,cy,cx,y,x):
    rad = math.atan2(y-cy,x-cx)
    distance = math.sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy))
    # 移動後の座標を出して返す
    _y = math.ceil(math.sin(rad+_rad) * distance + cy)
    _x = math.ceil(math.cos(rad+_rad) * distance + cx)
    return _y,_x
# 回転をいい感じに戻す処理部
def moldingrotation(_radian,center,bboxes):
    new_bboxes = []
    for bbox in bboxes:
        # デフォで何°ずれているのか計算
        y0,x0,y1,x1 = bbox
        # print("変換前 : ",bbox)
        # 各点に対して移動処理(8座標出現するかも)
        ay0,ax0 = molrot(_radian,center[0],center[1],y0,x0)
        ay1,ax1 = molrot(_radian,center[0],center[1],y0,x1)
        ay2,ax2 = molrot(_radian,center[0],center[1],y1,x0)
        ay3,ax3 = molrot(_radian,center[0],center[1],y1,x1)
        # これらを内包する最小矩形を錬成する
        miny = min(ay0,ay1,ay2,ay3)
        maxy = max(ay0,ay1,ay2,ay3)
        minx = min(ax0,ax1,ax2,ax3)
        maxx = max(ax0,ax1,ax2,ax3)
        new_bboxes.append([miny,minx,maxy,maxx])
        # print("返還後 : ","(",ay0,ax0,")","(",ay1,ax1,")","(",ay2,ax2,")","(",ay3,ax3,")")
        # print()
    # print()
    # print()
    return new_bboxes
# iou計算部分
def IoU(area1,area2,score = 0.2):
    y0,x0,y1,x1 = area1
    y2,x2,y3,x3 = area2
    # 片方図形内に頂点が存在できない場合に”重複しない”ことになる
    ymin,ymax = min(y0,y1),max(y0,y1)
    xmin,xmax = min(x0,x1),max(x0,x1)
    _ymin,_ymax = min(y2,y3),max(y2,y3)
    _xmin,_xmax = min(x2,x3),max(x2,x3)
    # yの内包状態
    y_inclusion = ((ymin<=y2 and y2<=ymax) or (ymin<=y3 and y3<=ymax)) or \
        ((_ymin<=y0 and y0<=_ymax) or (_ymin<=y1 and y1<=_ymax))
    x_inclusion = ((xmin<=x2 and x2<=xmax) or (xmin<=x3 and x3<=xmax)) or \
        ((_xmin<=x0 and x0<=_xmax) or (_xmin<=x1 and x1<=_xmax))
    if y_inclusion and x_inclusion:
        # 重複する場合
        AoO = (min(xmax,_xmax)-max(xmin,_xmin))*(min(ymax,_ymax)-max(_ymin,ymin))
        AoU = (xmax-xmin)*(ymax-ymin)+(_xmax-_xmin)*(_ymax-_ymin)-AoO
        return ((AoO/AoU) <= score)
    return True
    

# 重複削除部
def NMS(informations):
    # 重複を除いたものだけを返すものを作る
    bboxes,labels = [],[]
    next_queue = deque([])
    queue = deque(informations)
    res = []
    while len(queue) > 0:
        # まず先頭要素を基準に採る
        score,bbox,label = queue.popleft()
        bboxes.append(bbox)
        labels.append(label)
        # 残り要素をpopしながら比較する
        while len(queue) > 0:
            # 比較要素を採る
            _score,_bbox,_label = queue.popleft()
            # IoUを基準に、消すかどうか考える
            if IoU(bbox,_bbox):
                # 重複していない別要素なので残す
                next_queue.append([_score,_bbox,_label])
        # queue全部見終わったらqueue←next
        if len(next_queue) > 0:
            queue = next_queue
            next_queue = deque([])

    return bboxes,labels


def highclassinference(png):
    # 20°づつ回転させながら推論、その後元座標に全部変換し、重なった領域に対して最も信頼度が高いもののみを選ぶ
    # 回転させるのでまずは√2倍した下地を作る
    default_image = Image.open(png)
    w,h = default_image.size
    xy = math.ceil(w * math.sqrt(2))
    new_img = img=np.zeros((xy, xy, 3), np.uint8)
    # 貼り付ける
    img = img=np.zeros((xy, xy, 3), np.uint8)
    img[:,:,0:3]=[255,255,255]
    img = pil2cv(img)
    def_img = pil2cv(default_image)
    _xy = (xy-w)//2
    img[_xy:_xy+h,_xy:_xy+w] = def_img
    img = cv2pil(img)
    # 回転計算用変数の準備
    center = [xy/2,xy/2]

    # 学習済みmodelを渡す
    majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    model = SSD512(n_fg_class=len(majomoji_label))

    f_img = pilccv(img)
    # model のロード
    serializers.load_npz(model_name,model)
    informations = []

    runtime = time.time()
    jobs = []
    pipe_list = []
    # 18パターンの推論を回す処理
    for i in range(18):
        get_rev,send_rev  = multiprocessing.Pipe(False)
        deg = i*20
        rad = math.radians(deg)
        rot = img.rotate(deg,fillcolor=(255, 255, 255))
        rot = pilccv(rot)
        p = multiprocessing.Process(target=multi_run, args=(rot,model,send_rev,rad,center))
        jobs.append(p)
        pipe_list.append(get_rev)
        p.start()

        # /- 単一推論処理 -/
        # bboxes, labels, scores = run(rot,model)
        # for k in range(len(bboxes)):
        #     informations.append([scores[k],bboxes[k],labels[k]])

        # /- 角度毎の推論 -/
        # d_img = dwar_frame(bboxes,labels,f_img)
        # cv2.imwrite("rot_test"+str(i*20)+".png",d_img)
        # all_bboxes.append(bboxes)
        # all_labels.append(labels)
        
    
    # 受け取り判定
    for proc in jobs:
        proc.join()
    result_list = [x.recv() for x in pipe_list]
    # 受け取り後に成型
    for i in result_list:
        for k in range(len(i[0])):
            informations.append([i[2][k],i[0][k],i[1][k]])


    d_runtime = time.time() - runtime
    print("run : ",d_runtime," sec")
    nmstime = time.time()

    informations.sort(key=lambda x: x[0])
    bboxes,labels = NMS(informations)

    d_nmstime = time.time() - nmstime
    print("nms : ",d_nmstime," sec")

    img = pilccv(img)
    img = dwar_frame(bboxes,labels,img)
    # cv2.imwrite("iou_test_0.png",img)
    # discord用出力
    cv2.imwrite("fin_inf.jpg",img)
    print("finish")

def futures_inf(png):
    # 20°づつ回転させながら推論、その後元座標に全部変換し、重なった領域に対して最も信頼度が高いもののみを選ぶ
    # 回転させるのでまずは√2倍した下地を作る
    default_image = Image.open(png)
    w,h = default_image.size
    xy = math.ceil(w * math.sqrt(2))
    new_img = img=np.zeros((xy, xy, 3), np.uint8)
    # 貼り付ける
    img = img=np.zeros((xy, xy, 3), np.uint8)
    img[:,:,0:3]=[255,255,255]
    img = pil2cv(img)
    def_img = pil2cv(default_image)
    _xy = (xy-w)//2
    img[_xy:_xy+h,_xy:_xy+w] = def_img
    img = cv2pil(img)
    # 回転計算用変数の準備
    center = [xy/2,xy/2]

    # 学習済みmodelを渡す
    majomoji_label="A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    model = SSD512(n_fg_class=len(majomoji_label))

    f_img = pilccv(img)
    # model のロード
    serializers.load_npz(model_name,model)
    informations = []
    runtime = time.time()
    # 18パターンの画像を作る
    rot_img = []
    for i in range(18):
        deg = i*20
        rad = math.radians(deg)
        rot = img.rotate(deg,fillcolor=(255, 255, 255))
        rot = pilccv(rot)
        rot_img.append([rot,rad])
    # 並列実行
    with confu.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(multi_run2,x[0],model,x[1],center) for x in rot_img]
        (done, notdone) = confu.wait(futures)
        for future in confu.as_completed(futures):
            bboxes, labels, scores = future.result()
            for k in range(len(bboxes)):
                informations.append([scores[k],bboxes[k],labels[k]])
            


    d_runtime = time.time() - runtime
    print("run : ",d_runtime," sec")
    nmstime = time.time()

    informations.sort(key=lambda x: x[0])
    bboxes,labels = NMS(informations)

    d_nmstime = time.time() - nmstime
    print("nms : ",d_nmstime," sec")

    img = pilccv(img)
    img = dwar_frame(bboxes,labels,img)
    # cv2.imwrite("iou_test_0.png",img)
    # discord用出力
    cv2.imwrite("fin_inf.jpg",img)
    print("finish")


def test():
    a = 0
    f = 10000000*3
    for i in range(f):
        a += 1

def main():
    start = time.time()

    # test()

    futures_inf("84.PNG")

    d_time = time.time() - start
    print("total : ",d_time," sec")

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