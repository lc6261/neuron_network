"""
4.softmax
"""

#面向对象的神经网络类
import numpy as np

#激活函数
def activation_ReLU(input):
    return np.maximum(0, input)

def activation_softmax(inputs):
    
    """
    滑窗处理
    """
    # 计算输入矩阵inputs每一行的最大值，并保持原有的维度。这样做是为了避免数值上溢（exponential explosion）
    # 因为直接对原始输入应用指数函数可能导致非常大的数值。axis=1表示沿着行方向操作。
    max_values = np.max(inputs, axis=1, keepdims=True)
    print(max_values)

    #slided 滑窗:从输入中减去对应行的最大值，这一步称为平移。这样做不会改变最终的Softmax结果，但可以确保指数函数的输入不会太大，从而防止数值不稳定的问题。
    slided_inputs = inputs - max_values

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
    """
    归一化
    """
    #计算每行的指数值总和。这将是后续步骤中用来归一化各个值的基础。
    norm_base = np.sum(exp_values, axis=1, keepdims=True)

    #将每行的指数值除以它们各自的总和，得到的结果就是Softmax的概率分布。每个元素表示属于相应类别的概率，所有元素加起来等于1。
    norm_value = exp_values / norm_base


    return norm_value




#层类:layer
class Layer():
    def __init__(self, n_inputs, n_neurons):
        """
        初始定义三个变量
        :param n_inputs:输入端口数量
        :param n_neurons:网络节点数(神经元数)
        """
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.random.rand(self.n_inputs, self.n_neurons)#随机生成 n_inputs*n_neurons 的权重矩阵
        self.biases = np.random.rand(self.n_neurons)#随机生成 n_neurons的偏置

    # 前向计算函数
    def layer_forward(self,inputsData):
        """
        :param inputs:输入的参数(矩阵\向量)
        """
        sum1 = np.dot(inputsData, self.weights) + self.biases
        #激活函数
        #output = activation_ReLU(sum1)
        return sum1

    
#神经网络类:NetWork
class NetWork():
    #生成网络
    def __init__(self, network_shape):
        self.layers = []#初始化层
        self.network_shape = network_shape#初始化网络形状:[2, 3, 4, 5] ,数组单元个数是层数 ,每个下标对应每层的节点数

        for i in range(len(self.network_shape) - 1):#n层网络,输出(运算)n-1次,所以len(self.network_shape) - 1
            layer = Layer(network_shape[i], network_shape[i+1])#生成i层的节点,入参包括:1.输入端口数量 2.网络节点数
            self.layers.append(layer)

    #前馈运算
    def network_forward(self, inputsData):
        # 添加第一层的输入到队列
        outputs = [inputsData]
        print(f"输出第1层入参:\n{inputsData}")

        print(f"len(self.layers):{len(self.layers)}")
        
        for i in range(len(self.layers)):#n层网络,输出(运算)n-1次,
            layer_sum = self.layers[i].layer_forward(outputs[i])# 第i层输出结果

            if i < (len(self.layers)-1):
                layer_output = activation_ReLU(layer_sum)
            else:
                layer_output = activation_softmax(layer_sum)

            outputs.append(layer_output)# 添加i层的输出结果到队列
            print(f"输出第{i+1}层结果:\n{layer_output}")
            print("...........................")
        return outputs


def main():
    # 建立网络
    network = NetWork(NETWORK_SHAPE)
    # 前向计算
    network.network_forward(inputs)



#神经网络的形状:一共4层, 2, 3, 4, 2分别是各层的节点数
NETWORK_SHAPE = [2, 3, 4, 2]

#输入的数据
a11 = -0.9
a12 = 0.4

a21 = -0.8
a22 = 0.5

a31 = 0.5
a32 = -0.8

a41 = -0.6
a42 = -0.1

a51 = 0.5
a52 = 0.3

inputs = np.array([
    [a11, a12],
    [a21, a22],
    [a31, a32],
    [a41, a42],
    [a51, a52]
])

if __name__ == '__main__':
    main()