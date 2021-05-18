'''
设置好断点，debug运行，然后 F8 单步调试，
遇到想进入的函数 F7 进去，想出来在 shift + F8，
跳过不想看的地方，直接设置下一个断点，然后 F9 过去。
'''
import sys

import tensorflow as tf
from tqdm import tqdm
import numpy as np
from Client import Clients

'''
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
'''


def buildClients(num):
    learning_rate = 0.0001
    num_input = 28  # image shape: 32*32
    num_input_channel = 1  # image channel: 3
    num_classes = 10  # Cifar-10 total classes (0-9 digits)
    print("buildClients")
    # create Client and model
    return Clients(input_shape=[None, num_input, num_input, 1],
                   num_classes=num_classes,
                   learning_rate=learning_rate,
                   clients_num=num)


def run_global_test(client, global_vars, test_num):
    """ 跑一下测试集，输出ACC和Loss """
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))


# SOME TRAINING PARAMS ####
CLIENT_NUMBER = 10
# 每轮挑选clients跑跑看的比例
CLIENT_RATIO_PER_ROUND = 1.0
# epoch上限
epoch = 1

# CREATE CLIENT AND LOAD DATASET ####
client = buildClients(CLIENT_NUMBER)

# BEGIN TRAINING ####
# global_vars返回的是网络的结构
global_vars = client.get_client_vars()
noise_proportion_control = 0
clean_client_rate = 0.2
clean_client_num = CLIENT_NUMBER * clean_client_rate
clean_client_list = np.array(0, clean_client_num, 1)
for ep in range(epoch):
    # We are going to sum up active clients' vars at each epoch
    # 用来收集Clients端的参数，全部叠加起来（节约内存）
    client_vars_sum = None

    # Choose some clients that will train on this epoch
    # 随机挑选一些Clients进行训练
    # -参与训练的客户端，这里需要修改成自己选中的哪些客户端
    random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)

    # 用这些Clients进行训练，收集它们更新后的模型
    # tqdm进度条
    for client_id in tqdm(random_clients, ascii=True):
        # 将Server端的模型加载到Client模型上
        client.set_global_vars(global_vars)

        # train one client
        # 这个下标的Client进行训练
        client.train_epoch(cid=client_id,
                           noise_proportion=noise_proportion_control)

        # obtain current client's vars
        # 获取当前Client的模型变量值
        current_client_vars = client.get_client_vars()

        # sum it up
        # 把各个层的参数叠加起来
        # 这里不能直接累加起来，要改为依据条件生成权重---
        # 也就是说可以先用元组的方式保存起来，然后后面再分别读取修改
        if client_vars_sum is None:
            client_vars_sum = current_client_vars
        else:
            for cv, ccv in zip(client_vars_sum, current_client_vars):
                cv += ccv

    # obtain the avg vars as global vars
    # 把叠加后的Client端模型变量 除以 本轮参与训练的Clients数量
    # 得到平均模型、作为新一轮的Server端模型参数
    global_vars = []
    for var in client_vars_sum:
        global_vars.append(var / len(random_clients))

    # run test on 600 instances
    # 跑一下测试集、输出一下
    # run_test也可以自己设置为单独的
    run_global_test(client, global_vars, test_num=600)

# FINAL TEST#
run_global_test(client, global_vars, test_num=10000)

# outputfile.close()  # close后才能看到写入的数据
