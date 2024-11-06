"""
1.神经元
"""
import numpy as np


a1 = 0.9
a2 = 0.4
a3 = 0.7

#输入3个神经元 向量= 3列
inputs = np.array([
    [a1, a2, a3]
])

w1 = 0.8
w2 = 0.5
w3 = 0.6

#3个神经元 对应的3列权重 向量
weights = np.array([
    [w1], 
    [w2], 
    [w3]
])

#偏置向量 biases
b = -0.5


#点积
sum1 = np.dot(inputs, weights)

#激活函数,ReLU（Rectified Linear Unit）是一种常用的激活函数，用于神经网络中。它的输出是输入的线性部分，当输入为正数时，输出等于输入；当输入为负数时，输出等于零。
#ReLU 的数学表达式为：f(x) = max(0, x)
#ReLU 函数的优点包括：
#   非线性：ReLU 提供了非线性的特性，这对于解决复杂的机器学习问题至关重要。
#   计算效率高：相比于其他激活函数（如 sigmoid 和 tanh），ReLU 的计算速度更快，因为它只需要检查输入是否大于零。
#   梯度消失问题较少：在深度神经网络中，梯度消失问题是常见的问题之一。然而，由于 ReLU 在输入为正时保持不变，因此它有助于缓解这个问题。
def activation_ReLU(input):
    return np.maximum(0, input)

print(sum1)
print(f'sum1:{activation_ReLU(sum1)}')
