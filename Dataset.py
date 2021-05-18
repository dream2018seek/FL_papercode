from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.keras.utils import to_categorical
from function import load_data_mnist


class BatchGenerator:
    def __init__(self, x, yy):
        # 这里传入的是当前Cid拥有的所有train_data
        self.x = x
        self.y = yy
        # en(x)=5500
        self.size = len(x)
        print("BatchGenerator_self.size=", self.size)
        # 把len(x)长度作为一个从0开始的有顺序序列range(len(x))，并记录到list中打乱后作为索引
        self.random_order = list(range(len(x)))
        np.random.shuffle(self.random_order)
        self.start = 0
        return

    def next_batch(self, batch_size):
        perm = self.random_order[self.start:self.start + batch_size]

        self.start += batch_size
        if self.start > self.size:
            self.start = 0
        # print("next_batch-->self.y[perm].shape", self.y[perm].shape)
        # print("next_batch-->self.y[perm].type", type(self.y[perm]))

        return self.x[perm], self.y[perm]

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    # self.dataset = Dataset(tf.keras.datasets.cifar10.load_data, split=clients_num)
    # 构造函数带有默认参数
    def __init__(self, load_data_mnist, one_hot=True, split=0):
        (x_train, y_train), (x_test, y_test) = load_data_mnist()
        # print("Dataset: train-%d, test-%d" % (len(x_train), len(x_test)))

        '''
        if one_hot:
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)
        '''
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        # print('one_hot and astype(float32) end ')

        if split == 0:
            self.train = BatchGenerator(x_train, y_train)
        else:
            self.train = self.splited_batch(x_train, y_train, split)

        self.test = BatchGenerator(x_test, y_test)

    def splited_batch(self, x_data, y_data, split):
        res = []
        # 分割成split份，也即client_num份,并打包成一整个整体，记录为res,可通过res读取每个分割后的数据包
        for x, y in zip(np.split(x_data, split), np.split(y_data, split)):
            assert len(x) == len(y)
            # 这里res为打乱后的每个分割数据包的序列
            res.append(BatchGenerator(x, y))
        return res
