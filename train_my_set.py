import chainer
from chainer import training
from chainer.datasets import cifar
from chainer.datasets import mnist
import chainer.links as L
import chainer.functions as F
from chainer import iterators
#from chainercv.transforms import resize
from chainer.datasets import TransformDataset
from chainer import optimizers
from chainer.datasets import LabeledImageDataset
from chainer.training import extensions
import train_network as Net
import configparser

def train(network_object, batchsize=128, gpu_id=0, max_epoch=20, train_dataset=None, valid_dataset=None, test_dataset=None, postfix='', base_lr=0.01, lr_decay=None):

    # 1. Dataset

    # train = LabeledImageDataset('data/train/train_labels.txt', 'data/train/images')
    # valid = LabeledImageDataset('data/train/valid_labels.txt', 'data/valid/images')
    # #独自サイズに変更
    # def transform(in_data):
    #     img, label = in_data
    #     img = resize(img, (224, 224))
    #     return img, label
    # #image labelの順で入ってる
    # train = TransformDataset(train, transform)
    # valid = TransformDataset(valid, transform)

    # cifer10で学習(いちおう)
    if train_dataset is None and valid_dataset is None and test_dataset is None:
        train_val, test = cifar.get_cifar10()
        train_size = int(len(train_val) * 0.9)
        train, valid = chainer.datasets.split_dataset_random(train_val, train_size, seed=0)
    else:
        train, valid, test = train_dataset, valid_dataset, test_dataset

    # 2. Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    valid_iter = iterators.MultiprocessIterator(valid, batchsize, False, False)

    # 3. Model
    net = L.Classifier(network_object)

    # 4. Optimizer
    # SDGじゃなくて Adam使う
    # optimizer = optimizers.MomentumSGD(lr=base_lr).setup(net)
    optimizer = optimizers.Adam(lr=base_lr).setup(net)
    # 重みの発散を抑える
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # 5. Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

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
    
    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    trainer.run()
    del trainer

    # 8. Evaluation
    test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)
    test_evaluator = extensions.Evaluator(test_iter, net, device=gpu_id)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])

    return net

def main():
    # chainer.cuda.set_max_workspace_size(512*1024*1024)
    # #chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
    # chainer.config.autotune = True

    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'UTF-8')
    batchsize = ini['batchsize']
    max_epoch = ini['max_epoch']
    gpu_id = ini['gpu_id']
    postfix = ini['postfix']
    baseh_lr = ini['base_lr']
    lr_decay = ini['lr_decay']

    # ResNet 多分動くはず
    model = train(Net.ResNet(ResBlock), batch_size=batch_size , gpu_id=gpu_id, max_epoch=max_epoch, base_lr=base_lr, lr_decay=lr_decay)


if __name__ == "__main__":
    main()