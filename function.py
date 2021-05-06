from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# from functin import load_data_mnist
def load_data_mnist():
    """Loads CIFAR10 dataset.

    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # print('begin_load')
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = input_data.read_data_sets('location', one_hot=True)
    tf.logging.set_verbosity(old_v)
    # print('end_load')

    # print('''1)获得数据集的个数-begin''')
    train_nums = mnist.train.num_examples
    validation_nums = mnist.validation.num_examples
    test_nums = mnist.test.num_examples
    # print('MNIST数据集的个数')

    # print('''2)获得数据值-begin''')
    x_train = np.empty((train_nums, 3, 28, 28), dtype='uint8')
    y_train = np.empty((train_nums,), dtype='uint8')

    x_train = mnist.train.images  # 所有训练数据
    x_train = x_train.reshape([-1, 28, 28, 1]).astype('float32')
    # print('x_train_type:', type(x_train))
    # print('x_train_shape:', x_train.shape)
    val_data = mnist.validation.images  # (5000,784)
    x_test = mnist.test.images  # (10000,784)
    x_test = x_test.reshape([-1, 28, 28, 1]).astype('float32')
    # print('x_test_type:', type(x_test))
    # print('x_train_shape:', x_test.shape)

    # print('''3)获取标签值label=[0,0,...,0,1],是一个1*10的向量''')
    y_train = mnist.train.labels  # (55000,10)
    val_labels = mnist.validation.labels  # (5000,10)
    y_test = mnist.test.labels  # (10000,10)

    # print('array, newshape, order')
    # print('y_train.shape:', y_train.shape)
    # print('y_test.shape:', y_test.shape)
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
    # print('reshape x-test to x-train')
    x_test = x_train.astype(x_train.dtype)
    y_test = y_train.astype(y_train.dtype)
    # print('end')

    return (x_train, y_train), (x_test, y_test)