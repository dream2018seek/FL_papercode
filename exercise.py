from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import to_categorical

import os
import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export

from tensorflow.examples.tutorials.mnist import input_data


# 创建一个txt文件，文件名为mytxtfile
def text_create(name):
    desktop_path = "F:\\experiment\\federated learning\\FL-5\\output-log\\"

    # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')


filename = 'log'
text_create(filename)
output = sys.stdout
outputfile = open("F:\\experiment\\federated learning\\FL-5\\output-log\\" + filename + '.txt', 'w')
sys.stdout = outputfile

print('begin_load')
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('location', one_hot=True)
tf.logging.set_verbosity(old_v)
# print('end_load')

# print('''1)获得数据集的个数-begin''')
train_nums = mnist.train.num_examples
validation_nums = mnist.validation.num_examples
test_nums = mnist.test.num_examples
print('MNIST数据集的个数')

print('''2)获得数据值-begin''')
# x_train = np.empty((num_train_samples, 3, 32, 32), dtype='int8')
x_train = np.empty((train_nums, 3, 28, 28), dtype='uint8')
y_train = np.empty((train_nums,), dtype='uint8')

x_train = mnist.train.images  # 所有训练数据
print('x_train_type:', type(x_train))
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
x_train = x_train.reshape([-1, 28, 28, 1]).astype('float32')
print('x_train_shape:', x_train.shape, file=outputfile)
val_data = mnist.validation.images  # (5000,784)
x_test = mnist.test.images  # (10000,784)
x_test = x_test.reshape([-1, 28, 28, 1]).astype('float32')
# print('x_test_type:', type(x_test))
print('x_test_shape:', x_test.shape)

# print('''3)获取标签值label=[0,0,...,0,1],是一个1*10的向量''')
y_train = mnist.train.labels  # (55000,10)
val_labels = mnist.validation.labels  # (5000,10)
y_test = mnist.test.labels  # (10000,10)

print('array, newshape, order')
print('y_train.shape:', y_train.shape, file=outputfile)
print('y_test.shape:', y_test.shape, file=outputfile)
'''
y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

if K.image_data_format() == 'channels_last':
    # Theano和TensorFlow发生了分歧
    # channels_last:（样本数，行或称为高，列或称为宽，通道数）
    print('channels_last change')
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
'''
print('reshape x-test to x-train', file=outputfile)
x_test = x_train.astype(x_train.dtype)
y_test = y_train.astype(y_train.dtype)
print('end', file=outputfile)


outputfile.close()  # close后才能看到写入的数据
