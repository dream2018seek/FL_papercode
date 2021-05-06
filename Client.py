import tensorflow as tf
import numpy as np
from collections import namedtuple
import math

# 自定义的模型定义函数
from Model import AlexNet
# 自定义的数据集类
from Dataset import Dataset

# The definition of fed model
# 用namedtuple来储存一个模型
# train_op: tf计算图中的训练节点（一般是optimizer.minimize(xxx)）
# loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# optimizer = AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op')
# client = buildClients(CLIENT_NUMBER)
# class Clients:
#   def __init__(self, input_shape, num_classes, learning_rate, clients_num):
#   net = AlexNet(input_shape, num_classes, learning_rate, self.graph)
#   self.model = FedModel(*net)

from function import load_data_mnist


class Clients:
    # 初始化：在创建对象时候需要传入这些参数
    def __init__(self, input_shape, num_classes, learning_rate, clients_num):
        # 返回一个上下文管理器，这个上下管理器使用这个图作为要操作的图
        self.graph = tf.Graph()
        # tf.Session()创建一个会话，当上下文管理器退出时会话关闭和资源释放自动完成。
        self.sess = tf.Session(graph=self.graph)

        # 调用create函数来构建AlexNet的计算图
        # `net` 是一个list，依次包含模型中FedModel需要的计算节点（看上面）
        net = AlexNet(input_shape, num_classes, learning_rate, self.graph)
        print("net done")
        self.model = FedModel(*net)

        # 初始化
        # Variable：作为存储节点的变量（Variable）不是一个简单的节点，而是一副由四个子节点构成的子图：
        #         （1）变量的初始值——initial_value。
        #         （2）更新变量值的操作op——Assign。
        #         （3）读取变量值的操作op——read
        #         （4）变量操作——（a）
        # 上述四个步骤即：首先，将initial_value赋值（Assign）给节点，存储在（a）当中，当需要读取该变量时，调用read函数取值即可
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        # Load Cifar-10 dataset
        # NOTE: len(self.dataset.train) == clients_num
        # 加载数据集。对于训练集：`self.dataset.train[56]`可以获取56号client的数据集
        # `self.dataset.train[56].next_batch(32)`可以获取56号client的一个batch，大小为32
        # 对于测试集，所有client共用一个测试集，因此：
        # `self.dataset.test.next_batch(1000)`将获取大小为1000的数据集（无随机）
        # 这里已经把数据集划分成了多份，且每个client都有各自的一份
        self.dataset = Dataset(load_data_mnist,
                               split=clients_num)

    def run_test(self, num):
        """
            Predict the testing set, and report the acc and loss
            预测测试集，返回准确率和loss
            num: number of testing instances
        """
        with self.graph.as_default():
            batch_x, batch_y = self.dataset.test.next_batch(num)
            feed_dict = {
                self.model.X: batch_x,
                self.model.Y: batch_y,
                self.model.DROP_RATE: 0
            }
        return self.sess.run([self.model.acc_op, self.model.loss_op],
                             feed_dict=feed_dict)

    def train_epoch(self, cid, batch_size=32, dropout_rate=0.3):
        """
            Train one client with its own data for one epoch
            用`cid`号的client的数据对模型进行训练
            cid: Client id
        """
        dataset = self.dataset.train[cid]

        with self.graph.as_default():
            for _ in range(math.ceil(dataset.size / batch_size)):
                batch_x, batch_y = dataset.next_batch(batch_size)
                # print('batch_x.shape:%d, batch_y.shape:%s', batch_x.shape, batch_y.shape)
                feed_dict = {
                    self.model.X: batch_x,
                    self.model.Y: batch_y,
                    self.model.DROP_RATE: dropout_rate
                }
                self.sess.run(self.model.train_op, feed_dict=feed_dict)

    def get_client_vars(self):
        """ Return all of the variables list """
        # self.graph.as_default
        # 返回值：返回一个上下文管理器，这个上下管理器使用这个图作为默认的图
        with self.graph.as_default():
            # tf.trainable_variables()
            # 这个函数可以也仅可以查看可训练的变量，在我们生成变量时，
            # 无论是使用tf.Variable()还是tf.get_variable()生成变量，
            # 都会涉及一个参数trainable,其默认为True。
            client_vars = self.sess.run(tf.trainable_variables())
        return client_vars

    # 给客户端传入全局模型
    def set_global_vars(self, global_vars):
        # 为所有变量分配全局变量
        """ Assign all of the variables with global vars """
        with self.graph.as_default():
            # 返回使用 trainable=True 创建的所有变量
            all_vars = tf.trainable_variables()
            # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            # 也就是说通过使用zip将1个all_vars和1个global_vars打包成一个元组
            for variable, value in zip(all_vars, global_vars):
                variable.load(value, self.sess)

    def choose_clients(self, ratio=1.0):
        """
            randomly choose some clients
            随机选择`ratio`比例的clients，返回编号（也就是下标）
        """
        client_num = self.get_clients_num()
        # 返回大于等于参数x的最小整数,即对浮点数向上取整
        choose_num = math.ceil(client_num * ratio)
        # [:choose_num]从位置0到位置choose_num之前的数
        # 但放在这里表示为选取choose_num个数来随机打乱
        # 为什么要随机打乱呢
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        return len(self.dataset.train)
