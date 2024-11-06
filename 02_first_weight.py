"""
2.权重
"""

#实现一层神经网络layer1 = 2个神经元  3个输入
import numpy as np

a11 = -0.9
a12 = -0.4
a13 = -0.7

a21 = 0.8
a22 = 0.5
a23 = 0.6

a31 = -0.5
a32 = -0.8
a33 = -0.2


#batch ,生成3个输入值 ，批量处理
inputs = np.array([
    [a11, a12, a13],
    [a21, a22, a23],
    [a31, a32, a33]
])

# 生成随机数
def create_values(*args):
    result = np.random.rand(*args)
    return result

# 生成权重矩阵
def create_weight(n_inputs, n_neurons):
    result = create_values(n_inputs, n_neurons)#n_inputs 行 ,n_neurons 列
    return result

# 生成偏置值
def create_biases(n_nuerons):
    result = create_values(n_nuerons)
    return result

#激活函数
def activation_ReLU(input):
    return np.maximum(0, input)



#生成2个神经元的权重矩阵
print("生成2个神经元的 权重矩阵")
weights = create_weight(3, 2)
print(weights)

#生成2个神经元的偏置向量
print("生成2个神经元的 偏置向量")
b = create_biases(2)
print(b)

#点积
sum1 = np.dot(inputs, weights) + b

print("生成2个神经元的 前向传播值,激活函数前")
print(sum1)
print("生成2个神经元的 前向传播值,激活函数后")
print(activation_ReLU(sum1))
