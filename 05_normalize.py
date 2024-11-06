"""
5.归一化
"""

import numpy as np


def normalize(array):
    print("--------array--------------")
    print(array)

    """
    计算array中每一行的绝对值。
    """
    #绝对值
    absarray = np.absolute(array)

    """
    取每一行的最大值
    """
    # keepdims=True,axis=1表示沿着行方向操作。
    # 保证结果保持二维结构，即使只有一行或一列，这样方便后续与原始数组进行广播操作。
    max_nums = np.max(absarray, axis=1, keepdims=True)
    print("----------max_nums------------")
    print(max_nums)
    """
    计算每一行的缩放率
    """
    #使用np.where来创建一个缩放比率 scale_rate
    #如果某一行的最大绝对值是0（意味着这一行所有元素都是0），那么缩放比率为1（不做任何改变）。
    #否则，缩放比率是1除以最大绝对值，这样可以将最大绝对值变为1。也类似C的:scale_rate = (max_nums == 0)?1:1 / max_nums
    print("--------1 / max_nums--------------")
    print(1 / max_nums)
    
    #计算缩放率
    scale_rate = np.where(max_nums == 0, 1, 1 / max_nums)

    """
    每一行的缩放率*原始数据
    """
    #将原始数组array乘以相应的缩放比率scale_rate
    print("--------scale_rate--------------")
    print(scale_rate)
    norm = array * scale_rate

    print("--------normalize--------------")
    print(norm)

    return norm


#输入的数据
a11 = -0.9
a12 = 0.4

a21 = -0.8
a22 = 0.5

a31 = 0.5
a32 = -0.8

a41 = -0.001
a42 = -0.01

a51 = 0.5
a52 = 0.3

inputs = np.array([
    [a11, a12],
    [a21, a22],
    [a31, a32],
    [a41, a42],
    [a51, a52]
])
normalize(inputs)




def activation_softmax(inputs):
    
    print("--------inputs--------------")
    print(inputs)
    """
    滑窗处理
    """
    # 计算输入矩阵inputs每一行的最大值，并保持原有的维度。这样做是为了避免数值上溢（exponential explosion）
    # 因为直接对原始输入应用指数函数可能导致非常大的数值。axis=1表示沿着行方向操作。
    max_values = np.max(inputs, axis=1, keepdims=True)
    print("--------max_values--------------")
    print(max_values)

    #slided 滑窗:从输入中减去对应行的最大值，这一步称为平移。这样做不会改变最终的Softmax结果，但可以确保指数函数的输入不会太大，从而防止数值不稳定的问题。
    slided_inputs = inputs - max_values

    print("--------slided_inputs--------------")
    print(slided_inputs)
    """
    指数处理
    """
    #对平移后的数据应用指数函数。这样做的目的是为了放大差异性：正值会变得更大，负值会变得更小接近于零，而零保持不变。
    #np.exp() 是NumPy库提供的一个函数，用于计算自然对数的底e（约等于2.71828）的幂。对于数组或矩阵，它会对其中的每个元素分别应用指数函数。也就是说，如果slided_inputs是一个向量或矩阵，那么np.exp(slided_inputs)将返回一个新的向量或矩阵，其中的每个元素都是原对应位置元素的指数。
    #例如，如果slided_inputs包含这样的值：[-1, 0, 1]，那么np.exp(slided_inputs)的结果将是：
    #   e^(-1) ≈ 0.3679
    #   e^(0) = 1.0
    #   e^(1) ≈ 2.7183
    exp_values = np.exp(slided_inputs)
    print("--------exp_values--------------")
    print(exp_values)
    """
    归一化
    """
    #计算每行的指数值总和。这将是后续步骤中用来归一化各个值的基础。
    norm_base = np.sum(exp_values, axis=1, keepdims=True)
    
    print("--------norm_base--------------")
    print(norm_base)
    #将每行的指数值除以它们各自的总和，得到的结果就是Softmax的概率分布。每个元素表示属于相应类别的概率，所有元素加起来等于1。
    norm_value = exp_values / norm_base

    print("--------norm_value--------------")
    print(norm_value)

    return norm_value

activation_softmax(inputs)
