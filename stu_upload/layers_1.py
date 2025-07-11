# coding=utf-8
import numpy as np
import struct
import os
import time

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    def forward(self, input):  # 前向传播计算
        start_time = time.time()
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果
        self.output = self.input @ self.weight + self.bias
        return self.output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        self.d_weight = self.input.T @ top_diff
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff =  top_diff @ self.weight.T
        return bottom_diff
    def update_param(self, lr):  # 参数更新

            # --- 加入这些调试打印语句 ---
        # np.linalg.norm 用来计算矩阵或向量的模
        # 如果模是0，说明整个梯度矩阵/向量都是0
        d_weight_norm = np.linalg.norm(self.d_weight)
        d_bias_norm = np.linalg.norm(self.d_bias)

        # 为了避免打印太多信息，我们可以只打印第一个全连接层(fc1)的梯度信息
        # fc1的输入维度是784，以此作为判断条件
        if self.num_input == 784:
            print(f"Updating fc1: d_weight_norm={d_weight_norm:.6f}, d_bias_norm={d_bias_norm:.6f}")
        # ---------------------------



        # TODO：对全连接层参数利用参数进行更新
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):  # 参数保存
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        output = np.maximum(0, input)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff

class SoftmaxLossLayer(object): ##要把损失函数伪装成普通的层来方便
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):  # 前向传播的计算 ---> 计算不用于直接输出的概率分布
        # TODO：softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max) #计算关于底数e的指数
        self.prob = input_exp / np.sum(input_exp, axis = 1, keepdims = True)    #预测概率
        return self.prob
    def get_loss(self, label):   # 计算损失  --->  在工作代码中调用(?)
        self.batch_size = self.prob.shape[0]    #用于平均化到和为1的除以值(?)
        self.label_onehot = np.zeros_like(self.prob)    #真实标签 P_i(k)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算  --->  用于输出损失值反向传播给隐层
        # TODO：softmax 损失层的反向传播，计算本层损失
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

