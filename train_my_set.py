from chainer.datasets import mnist

train_val, test = mnist.get_mnist(withlabel=True, ndim=1)


#自作データセット読み込み
from chainer.datasets import LabeledImageDataset
train = LabeledImageDataset('data/train/train_labels.txt', 'data/train/images')



def main():






#?????
if __name__ == '__main__':
    main()