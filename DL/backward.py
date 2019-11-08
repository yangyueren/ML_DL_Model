#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
A module to implement the stochastic gradient descent learning algorithm
for a feedforward neural network.
"""

import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        """
        :param sizes: list, 存储每层的神经元数目sizes = [2,3,4]
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        前向传输计算每个神经元的值
        :param a: 输入值
        :return: 计算后每个神经元的值
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, traing_data, epochs, mini_batch_size, eta, test_data=None):
        """
        随机梯度下降
        :param traing_data: 输入的训练集
        :param epochs: 迭代次数
        :param mini_batch_size: 小样本数量
        :param eta: 学习率
        :param test_data: 测试数据集
        :return:
        """
        if test_data:
            n_test = len(test_data)
        n = len(traing_data)
        for j in range(epochs):
            # 搅乱训练集
            random.shuffle(traing_data)
            mini_batchs = [traing_data[k:k+mini_batch_size]
                           for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)}, {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """
        更新w和b的值
        :param mini_batch:
        :param eta:
        :return:
        """
        # [2,3,4]
        #biases [3,4]
        #weights [(2,3),(3,4)]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self,x, y):
        """

        :param x:
        :param y:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #前向传输
        activation = x
        #存储每层神经元的值的矩阵
        activations = [x]
        #存储每个未经过sigmoid计算的神经元的值
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def cost_derivate(self, output_activations, y):
        return (output_activations-y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

def sigmoid(x):
    """
    sigmoid function
    :param x:
    :return:
    """
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    """
    sigmoid函数的导数
    :param x:
    :return:
    """
    return sigmoid(x)*(1-sigmoid(x))