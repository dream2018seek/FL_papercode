# Model.py：定义TF模型的计算图

import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.train import AdamOptimizer


#### Create tf model for Client ####

def AlexNet(input_shape, num_classes, learning_rate, graph):
    """
        Construct the AlexNet model.
        input_shape: The shape of input (`list` like)
        num_classes: The number of output classes (`int`)
        learning_rate: learning rate for optimizer (`float`)
        graph: The tf computation graph (`tf.Graph`)
    """
    with graph.as_default():
        X = tf.placeholder(tf.float32, input_shape, name='X')
        Y = tf.placeholder(tf.float32, [None, 10], name='Y')
        print("net___input_shape=", input_shape)
        DROP_RATE = tf.placeholder(tf.float32, name='drop_rate')

        X = tf.reshape(X, [-1, 28, 28, 1])

        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        # conv1 = conv(X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        # conv include Add biases   Apply relu function
        # def conv(x, filter_height, filter_width, num_filters,
        #        stride_y, stride_x, name, padding='SAME', groups=1):
        conv1 = conv(X, 5, 5, 20, 1, 1, padding='VALID', name='conv1')
        # norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        # pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        # conv2 = conv(norm1, 5, 5, 64, 1, 1, groups=2, name='conv2')
        conv2 = conv(conv1, 5, 5, 50, 1, 1, padding='VALID', name='conv2')
        # norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')

        # 论文里写为平均池化
        pool3 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name='pool3')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        # flattened = tf.reshape(pool5, [-1, 6*6*256])
        # fc6 = fc(flattened, 6*6*256, 4096, name='fc6')

        # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        # flattened = tf.reshape(pool3, [-1, 1 * 1 * 256])
        flattened = tf.reshape(pool3, [-1, 10 * 10 * 50])
        # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        # fc_layer(x, input_size, output_size, name, relu=true)
        fc4 = fc_layer(flattened, 10 * 10 * 50, 256, name='fc4')
        dropout4 = dropout(fc4, DROP_RATE)

        # 7th Layer: FC (w ReLu) -> Dropout
        # fc7 = fc(dropout6, 4096, 4096, name='fc7')
        # fc5 = fc_layer(dropout4, 1024, 2048, name='fc5')
        # dropout5 = dropout(fc5, DROP_RATE)

        # 8th Layer: FC and return unscaled activations
        logits = fc_layer(dropout4, 256, num_classes, relu=False, name='fc5')

        # loss and optimizer
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                       labels=Y))
        optimizer = AdamOptimizer(
            learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model
        prediction = tf.nn.softmax(logits)
        pred = tf.argmax(prediction, 1)

        # accuracy
        correct_pred = tf.equal(pred, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32))

        return X, Y, DROP_RATE, train_op, loss_op, accuracy


def conv(x, filter_height, filter_width, num_filters,
         stride_y, stride_x, name, padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    print("def conv___input_channels=", input_channels)

    # Create lambda function for the convolution
    # 创建卷积函数
    # tf.nn.conv2d(输入描述，卷积核描述，核滑动步长（1，行步长，列步长，1），padding)
    # Input： [ batch, in_height, in_width, in_channel ]
    # filter：[ filter_height, filter_width, in_channel, out_channels ]
    convolve = lambda i, k: tf.nn.conv2d(
        i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    # 创建全局可以使用的变量模型参数和偏置
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        # weight是存放中间过程的模型参数，shape为（卷积核大小、）
        # <tf.Variable 'conv1/weights:0' shape=(5, 5, 1, 20) dtype=float32_ref>
        weights = tf.get_variable('weights',
                                  shape=[
                                      filter_height, filter_width,
                                      input_channels / groups, num_filters
                                  ])
        # shape=(20,)
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        # conv1 = conv(X, 5, 5, 20, 1, 1, padding='VALID', name='conv1')
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3,
                                 num_or_size_splits=groups,
                                 value=weights)
        output_groups = [
            convolve(i, k) for i, k in zip(input_groups, weight_groups)
        ]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


# fc4 = fc_layer(flattened, 1 * 1 * 256, 1024, name='fc4')
# 最后全连接的前一层还是有在进行池化的
def fc_layer(x, input_size, output_size, name, relu=True, k=20):
    """Create a fully connected layer."""

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases.
        W = tf.get_variable('weights', shape=[input_size, output_size])
        b = tf.get_variable('biases', shape=[output_size])
        # Matrix multiply weights and inputs and add biases.
        z = tf.nn.bias_add(tf.matmul(x, W), b, name=scope.name)

    if relu:
        # Apply ReLu non linearity.
        a = tf.nn.relu(z)
        return a

    else:
        return z


def max_pool(x,
             filter_height, filter_width,
             stride_y, stride_x,
             name, padding='SAME'):
    """Create a max pooling layer."""
    # print("def_max_pool---->ksize",filter_width, filter_height)
    # print("def_max_pool---->stride", stride_y, stride_x)
    return tf.nn.avg_pool2d(x,
                            ksize=[1, filter_height, filter_width, 1],
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding,
                            name=name)
    # return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # pool3 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool3')

# p05 = tf.nn.avg_pool2d(conv10,
#                        ksize=[1,conv10.get_shape().as_list()[1],conv10.get_shape().as_list()[1],1],
#                        strides=[1,1,1,1],padding=VALID',name='GAP')


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)


def dropout(x, rate):
    """Create a dropout layer."""
    return tf.nn.dropout(x, rate=rate)
