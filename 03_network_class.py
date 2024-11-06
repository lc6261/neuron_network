"""
3.面向对象的神经网络类
"""
import numpy as np

#激活函数
def activation_ReLU(input):
    return np.maximum(0, input)


#神经网络的形状:一共4层, 2, 3, 4, 2分别是各层的节点数
NETWORK_SHAPE = [2, 3, 4, 2]

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
        output = activation_ReLU(sum1)
        return output

    
#神经网络类:NetWork
class NetWork():
    #生成网络
    def __init__(self, network_shape):
        self.layers = []#初始化层
        self.network_shape = network_shape#初始化网络形状:[2, 3, 4, 5] ,数组单元个数是层数 ,每个下标对应每层的节点数

        for i in range(len(self.network_shape) - 1):#n层网络,输出(运算)n-1次,所以len(self.network_shape) - 1
            layer = Layer(network_shape[i], network_shape[i+1])#生成i层的节点,入参包括:1.输入端口数量 2.网络节点数
            self.layers.append(layer)

    #神经网络的推理(或称为前向传播)
    def network_forward(self, inputsData):
        # 添加第一层的输入到队列
        outputs = [inputsData]
        print(f"输出第1层入参:\n{inputsData}")

        print(f"len(self.layers):{len(self.layers)}")
        
        for i in range(len(self.layers)):#n层网络,输出(运算)n-1次,
            layer_output = self.layers[i].layer_forward(outputs[i])# 第i层输出结果
            outputs.append(layer_output)# 添加i层的输出结果到队列
            print(f"输出第{i+1}层结果:\n{layer_output}")
            print("...........................")
        return outputs


def main():
    # 建立网络
    network = NetWork(NETWORK_SHAPE)
    # 前向计算
    network.network_forward(inputs)



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