"""
7.损失函数(损失函数返回值越小，表示预测值越接近实际值)
def precise_loss_func(predicted, real):

损失函数是机器学习和深度学习中的一个重要概念。它用来衡量模型的预测结果与实际目标值之间的差距。损失函数的主要作用包括:

1. 评估模型性能:通过计算损失值,我们可以知道模型的预测有多准确。

2. 指导模型优化:在训练过程中,我们的目标是最小化损失函数。模型通过不断调整参数来降低损失值,从而提高预测准确性。

3. 比较不同模型:使用相同的损失函数可以帮助我们比较不同模型的性能。

常见的损失函数包括:

- 均方误差(MSE):用于回归问题,计算预测值与实际值之差的平方。
- 交叉熵损失:常用于分类问题,特别是在神经网络中。
- Hinge损失:用于支持向量机(SVM)等分类器。

选择合适的损失函数对于模型的训练和性能至关重要。不同的问题类型和数据特征可能需要不同的损失函数。

总之,损失函数是机器学习中连接模型预测和实际目标的桥梁,在整个学习过程中起着核心作用。


"""
import numpy as np
import importlib
cp = importlib.import_module("06_createdata_plot")



# 损失函数
def precise_loss_func(predicted, real):
    """
    计算预测值和实际值之间的精确损失。

    - 当预测完全错误时（预测概率与实际类别完全不匹配），损失接近1
    - 当预测完全正确时（预测概率与实际类别完全匹配），损失接近0

    具体计算步骤如下:
    1. 将实际值转换为one-hot编码矩阵
        （one-hot编码矩阵是一种将类别变量转换为二进制向量的方法，其中只有一个元素为1，其余元素为0。
        例如，对于二分类问题，类别0可以表示为[1,0]，类别1可以表示为[0,1]）
    2. 计算预测值和实际值的点积
    3. 对点积结果求和
    4. 用1减去求和结果得到最终损失
    
    参数:
    predicted: 预测的概率分布，形状为 (n_samples, n_classes)
    real: 实际的类别标签，形状为 (n_samples,)
    
    返回:
    每个样本的损失值，形状为 (n_samples,)
    """
    # 打印预测值
    print(f"predicted:\n{predicted}")
    
    # 创建一个形状为 (len(real), 2) 的零矩阵
    #类别0: [1, 0]
    #类别1: [0, 1]
    
    # 创建一个形状为 (len(real), 3) 的零矩阵
    #类别0: [1, 0, 0]
    #类别1: [0, 1, 0]
    #类别2: [0, 0, 1]
    real_matrix = np.zeros((len(real), 2))
    
    # 将实际值填充到矩阵的第二列
    # 这里假设 real 中的值为 0 或 1，分别代表两个类别
    # 对于类别 1，第二列将为 1；对于类别 0，第二列将为 0
    real_matrix[:,1] = real
    
    # 将 1 减去实际值填充到矩阵的第一列
    # 这样可以得到 one-hot 编码的互补值
    # 对于类别 1，第一列将为 0；对于类别 0，第一列将为 1
    # 这样就完成了 one-hot 编码的转换：[1,0] 表示类别 0，[0,1] 表示类别 1
    real_matrix[:,0] = 1 - real
    
    # 打印转换后的实际值矩阵
    print(f"real_matrix-bit:\n{real_matrix}")
    
    # 计算预测值和实际值的点积，得到每个样本的损失
    # 这里使用 `*` 是正确的，因为我们想要对应位置的元素相乘，然后沿着行求和。这正好符合损失函数的计算需求。
    product = np.sum(predicted * real_matrix, axis=1)
    
    # 返回每个样本的损失值
    return 1-product


def activation_ReLU(input):
    return np.maximum(0, input)

def activation_softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)
    print(max_values)
    slided_inputs = inputs - max_values
    exp_values = np.exp(slided_inputs)
    norm_base = np.sum(exp_values, axis=1, keepdims=True)
    norm_value = exp_values / norm_base
    return norm_value

def classify(input_probabilities):
    return np.rint(input_probabilities[:,1])

NETWORK_SHAPE = [2, 3, 4,5, 2]

class Layer():
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.random.rand(self.n_inputs, self.n_neurons)
        self.biases = np.random.rand(self.n_neurons)

    def layer_forward(self,inputsData):
        sum1 = np.dot(inputsData, self.weights) + self.biases
        return sum1

class NetWork():
    def __init__(self, network_shape):
        self.layers = []
        self.network_shape = network_shape

        for i in range(len(self.network_shape) - 1):
            layer = Layer(network_shape[i], network_shape[i+1])
            self.layers.append(layer)

    def network_forward(self, inputsData):
        outputs = [inputsData]
        print(f"输出第1层入参:\n{inputsData}")

        print(f"len(self.layers):{len(self.layers)}")
        
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])

            if i < (len(self.layers)-1):
                layer_output = activation_ReLU(layer_sum)
            else:
                layer_output = activation_softmax(layer_sum)

            outputs.append(layer_output)
            print(f"输出第{i+1}层结果:\n{layer_output}")
            print("...........................")
            
        return outputs

def main():
    # 创建数据点
    data = cp.creat_data(10)

    # 绘制原始数据
    cp.plot_data(data,"orgin")
    
    # 从data数组中选择所有行的前两列作为输入特征
    # 这里假设data的前两列包含了我们需要的特征数据
    # 例如，如果data是一个包含坐标和标签的数组，前两列可能是x和y坐标
    inputs = data[:,(0,1)]

    # 使用预定义的网络形状NETWORK_SHAPE初始化一个新的神经网络实例
    # NETWORK_SHAPE定义了网络的层数和每层的神经元数量
    network = NetWork(NETWORK_SHAPE)

    # 使用神经网络对输入数据进行前向传播计算
    # 返回的outputs包含了每一层的输出结果
    outputs = network.network_forward(inputs)

    # 对神经网络的最后一层输出进行分类
    # outputs[-1] 表示神经网络最后一层的输出
    # classify 函数将概率值转换为二元分类结果
    classifydata = classify(outputs[-1])
    
    print(f"分类结果:\n{classifydata}")

    # 将神经网络的分类结果赋值给data数组的第3列（索引为2）
    # 这样做可以将原始数据和分类结果合并在一起
    data[:,2]= classifydata
    
    print(f"原始数据+分类结果:\n{data}")
    
    # 计算损失
    # 参数说明:
    # outputs[-1]: 神经网络最后一层的输出，即预测值
    # data[:,2]: 原始数据的第三列，即实际标签值
    # precise_loss_func: 自定义的精确损失函数
    # 返回值: 每个样本的损失值
    precise_loss = precise_loss_func(outputs[-1], data[:,2])
    print(f"损失:\n{precise_loss}")

    # 绘制训练前的分类结果
    cp.plot_data(data,"before training")

if __name__ == '__main__':
    main()
