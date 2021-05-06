
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export


dirname = 'cifar-10-batches-py'
origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

path = get_file(
    dirname,
    origin=origin,
    untar=True,
    file_hash=
    '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce')

num_train_samples = 50000

# empty(shape[, dtype, order])
x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
y_train = np.empty((num_train_samples,), dtype='uint8')

for i in range(1, 6):
    # 如果参数中某个部分是绝对路径，则绝对路径前的路径都将被丢弃，并从绝对路径部分开始连接。
    # load_batch return:
    # data = data.reshape(data.shape[0], 3, 32, 32)
    # labels = d[label_key]
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
print('x_train.type:', type(x_train))
print('y_train.type:', type(y_train))
print('x_train.shape:', x_train.shape)
print('y_train.shape:', np.shape(y_train))

fpath = os.path.join(path, 'test_batch')
x_test, y_test = load_batch(fpath)
print('x_test.shape:', x_test.shape)
print('y_test.shape:', np.shape(y_test))

y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))
print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)

if K.image_data_format() == 'channels_last':
    # -transpose函数作用就是调换数组的行列值的索引值
    # -这里是第一维度不变，剩下的三个维度变化
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

x_test = x_test.astype(x_train.dtype)
y_test = y_test.astype(y_train.dtype)
print('x_train.shape:', x_train.shape)
print('y_train.shape:', y_train.shape)
print('x_test.shape:', x_test.shape)
print('y_test.shape:', y_test.shape)