import numpy as np
import importlib
cp = importlib.import_module("06_createdata_plot")


"""
逻辑解释
目标值矩阵的构建:
target 矩阵的第一列存储的是 1 - target_vector ，即如果实际目标是0，则第一列为1；如果实际目标是1，则第一列为0。
target 矩阵的第二列直接存储 target_vector。
置信度计算:
对于每个样本，计算 target[i] 和 predicted_values[i] 的点积 confidence。
点积的结果反映了预测值与实际目标的一致性。如果点积结果大于0.5，说明预测较为准确，不需要调整；否则，需要调整。
调整规则:
如果 confidence > 0.5，则将 target[i] 设为 [0, 0]，表示不需要任何调整。
如果 confidence <= 0.5，则将 target[i] 进行调整，公式为 (target[i] - 0.5) * 2。这会将 [0, 1] 范围内的值映射到 [-1, 1] 范围内：
原来的0变为-1，表示需要减少。
原来的1变为1，表示需要增加。
原来的0.5变为0，表示不需要改变。
"""
def adjust_final_layer_targets(predicted_values, target_vector):
    """
    调整最后一层的目标值

    参数:
    predicted_values: 预测值，形状为 (n_samples, 2)
    target_vector: 目标向量，形状为 (n_samples,)

    返回:
    调整后的目标值，形状为 (n_samples, 2) 
        第一列是1减去目标值,第二列是目标值，
        如果是0,不需要调整
            是1,需要增加
            是-1,需要减少
    """
    #print(f"predicted_values:\n{predicted_values}")
    print(f"target_vector:\n{target_vector}")
    
    # 创建一个二维数组来存储目标值
    # 创建一个形状为(样本数量, 2)的零矩阵来存储目标值
    # 第一列将存储1减去目标值，第二列将存储原始目标值
    target = np.zeros((len(target_vector), 2))
    
    # 将目标向量填充到第二列
    # 这里使用切片操作将target_vector赋值给target矩阵的第二列（索引为1）
    # 这样做可以保持原始目标值在矩阵的第二列
    target[:, 1] = target_vector
    
    # 将1减去目标向量填充到第一列
    target[:, 0] = 1 - target_vector

    for i in range(len(target_vector)):
        # 计算置信度：目标值和预测值的点积
        # 这里使用numpy的点积运算，相当于 target[i][0]*predicted_values[i][0] + target[i][1]*predicted_values[i][1]
        # 如果预测正确，置信度应该接近1；如果预测错误，置信度应该接近0
        # 使用点积计算置信度
        # 点积可以有效地计算两个向量的相似度
        # 在这里,它用于比较目标值和预测值的匹配程度
        # 如果两个向量方向一致(即预测正确),点积结果会较大
        # 如果方向相反(即预测错误),点积结果会较小
        # 这种方法比单纯比较数值更能反映预测的准确性
        confidence = np.dot(target[i], predicted_values[i])
        
        # 如果置信度大于0.5,说明预测正确,不需要调整
        # 将目标值设为[0, 0],表示不需要任何改变
        if confidence > 0.5:
            target[i] = np.array([0, 0])
        else:
            # 如果置信度不大于0.5,说明预测不够准确,需要调整目标值
            # 将目标值减去0.5,然后乘以2,可以将[0,1]范围的值映射到[-1,1]范围
            # 这样做可以增强调整的效果:
            # - 原来接近0的值会变成负数,表示需要减小
            # - 原来接近1的值会变成正数,表示需要增加
            # - 原来接近0.5的值变化较小,表示需要轻微调整
            target[i] = (target[i] - 0.5) * 2

    # 返回调整后的目标值
    return target

def precise_loss_func(predicted, real):
    print(f"predicted:\n{predicted}")
    
    real_matrix = np.zeros((len(real), 2))
    
    real_matrix[:,1] = real
    
    real_matrix[:,0] = 1 - real
    
    print(f"real_matrix-bit:\n{real_matrix}")
    
    product = np.sum(predicted * real_matrix, axis=1)
    
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

NETWORK_SHAPE = [2, 3, 4, 5, 2]

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
    
    # 创建目标值的深拷贝
    # 使用np.copy()函数复制data数组的第3列（索引为2）
    # 这样可以避免直接引用原始数据，防止后续操作意外修改原始数据
    targets = np.copy(data[:,2])


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
    #data[:,2]= classifydata
    
    #print(f"原始数据+分类结果:\n{data}")
    
    
    print(f"原始标签:\n{data[:,2]}")
    # 计算损失
    # 参数说明:
    # outputs[-1]: 神经网络最后一层的输出，即预测值
    # data[:,2]: 原始数据的第三列，即实际标签值
    # precise_loss_func: 自定义的精确损失函数
    # 返回值: 每个样本的损失值
    precise_loss = precise_loss_func(outputs[-1], data[:,2])
    print(f"损失(接近1表示预测错误):\n{precise_loss}")

    # 绘制训练前的分类结果
    cp.plot_data(data,"before training")

    # 调整最后一层的目标值
    # outputs[-1] 是网络的最终输出
    # data[:,2] 是真实的标签值
    # 这个函数会计算出需要对最后一层进行的调整
    target_values = adjust_final_layer_targets(outputs[-1], targets)
    
    print(f"需求函数调整后的目标值:\n{target_values}")


if __name__ == '__main__':
    main()
