from chainer.datasets import mnist
import chainer.links as L
import chainer.functions as F
from chainer import iterators
from chainercv.transforms import resize
from chainer.datasets import TransformDataset
from chainer import optimizers
from chainer.datasets import LabeledImageDataset
import train_network as ResNet

def train(network_object, batchsize=128, gpu_id=0, max_epoch=20, train_dataset=None, valid_dataset=None, test_dataset=None, postfix='', base_lr=0.01, lr_decay=None):

    # 1. Dataset

    train = LabeledImageDataset('data/train/train_labels.txt', 'data/train/images')
    valid = LabeledImageDataset('data/train/valid_labels.txt', 'data/valid/images')
    #独自サイズに変更
    def transform(in_data):
        img, label = in_data
        img = resize(img, (224, 224))
        return img, label
    #image labelの順で入ってる
    train = TransformDataset(train, transform)
    valid = TransformDataset(valid, transform)

    # 2. Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    valid_iter = iterators.MultiprocessIterator(valid, batchsize, False, False)

    # 3. Model
    net = L.Classifier(network_object)

    # 4. Optimizer
    optimizer = optimizers.MomentumSGD(lr=base_lr).setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # 5. Updater
    updater = training.StasndardUpdater(train_iter, optimizer, device=gpu_id)

    # 6. Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_cifar10_{}result'.format(network_object.__class__.__name__, postfix))

    # 7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    if lr_decay is not None:
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_decay)
    trainer.run()
    del trainer

    # 8. Evaluation
    test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)
    test_evaluator = extensions.Evaluator(test_iter, net, device=gpu_id)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])

    return net

def main():
    chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
    chainer.config.autotune = True
    model = train(ResNet(10), max_epoch=100, base_lr=0.1, lr_decay=(30, 'epoch'))


if __name__ == "__main__":
    main()