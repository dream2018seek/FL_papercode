from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.train import AdamOptimizer


# 初始化权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置项
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


# 定义一个2*2的最大池化层
def max_pool_2_2(x):
    return tf.nn.avg_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


if __name__ == "__main__":
    # 定义输入变量
    x = tf.placeholder("float", shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    print("input_shape=", x.shape)
    # 定义输出变量
    y_ = tf.placeholder("float", shape=[None, 10])

    # 第一层卷积
    w_conv1 = weight_variable([5, 5, 1, 20])
    b_conv1 = bias_variable([20])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

    # 第二层卷积
    w_conv2 = weight_variable([5, 5, 20, 50])
    b_conv2 = bias_variable([50])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2) + b_conv2)

    # 第三层：池化层
    h_pool = max_pool_2_2(h_conv2)
    print("h_pool_shape=", h_pool.shape)

    # 第四层：全连接层
    w_fc1 = weight_variable([10 * 10 * 50, 256])
    b_fc1 = bias_variable([256])
    h_pool_flat = tf.reshape(h_pool, [-1, 10 * 10 * 50])
    print("h_pool_flat_shape=", h_pool_flat.shape)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, w_fc1) + b_fc1)
    keep_prob = tf.placeholder("float32")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 第五层：输出层
    w_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    # 最小化交叉熵损失
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv,
                                                   labels=y_))
    optimizer = AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    prediction = tf.nn.softmax(y_)
    pred = tf.argmax(prediction, 1)
    # 计算准确率：一样返回True,否则返回False
    correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess = tf.Session()
    # tf.global_variables_initializer
    writer = tf.summary.FileWriter('./mylogs', sess.graph)
    sess.run(tf.initialize_all_variables())

    # 下载mnist的手写数字的数据集
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    loss_summary = tf.summary.scalar('loss', loss_op)
    acc_summary = tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge([loss_summary, acc_summary])
    # merged = tf.summary.merge_all()

    # 日志输出，每迭代100次输出一次日志
    for i in range(801):
        batch = mnist.train.next_batch(128)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d,training accuracy %g" % (i, train_accuracy))
        train_op.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # 在执行sess.run()时，tensorflow并不是计算了整个图，只是计算了与想要fetch的值相关的部分。
        summary = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        writer.add_summary(summary, i)

    print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # 用cmd: tensorboard --logdir=F:\MyFL\federated_learning\FL-5\tools\tensorboard-learn\tensorboard\mylogs
    # http://URI70Z6NP5RWLP7:6006/
    # (keras-gpu) F:\MyFL>cd federated_learning
    # cd FL-5
    # cd tools
    # cd tensorboard-learn
