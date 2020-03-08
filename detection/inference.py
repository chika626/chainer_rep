import matplotlib.pyplot as plt
from chainer import serializers
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD512
from chainercv.links import SSD300
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox

def main():
    # 推論させたい画像の選択
    img = read_image('sample.jpg')

    # 学習済みmodelを渡す
    # model = FasterRCNNVGG16(pretrained_model='voc0712')
    model = SSD512(n_fg_class=len(voc_bbox_label_names))
    # model のロード
    serializers.load_npz('my_ssd_model.npz',model)

    # 推論の実行？
    bboxes, labels, scores = model.predict([img])
    vis_bbox(img, bboxes[0], labels[0], scores[0],
         label_names=voc_bbox_label_names)

    # 4座標はバウンディングボックスに入る
    # これを分類にかければ精度出そう
    print(bboxes[0])
    plt.show()


if __name__ == '__main__':
    main()
