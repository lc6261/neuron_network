import numpy as np
import importlib
cp = importlib.import_module("06_createdata_plot")

def normalize(inputarray):
    max_values = np.max(np.absolute(inputarray),axis=1,keepdims=True)
    scaled_rate = np.where(max_values == 0, 1, 1 / max_values)
    norm_data = inputarray * scaled_rate
    return norm_data
    
def adjust_final_layer_targets(predicted_values, target_vector):
    print(f"target_vector:\n{target_vector}")
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector
    for i in range(len(target_vector)):
        confidence = np.dot(target[i], predicted_values[i])
        
        if confidence > 0.5:
            target[i] = np.array([0, 0])
        else:
            target[i] = (target[i] - 0.5) * 2
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

class Layer():
    def __init__(self, n_inputs, n_neurons, layer_number):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.layer_number = layer_number
        self.weights = np.random.rand(self.n_inputs, self.n_neurons)
        self.biases = np.random.rand(self.n_neurons)
    def layer_forward(self,inputsData):
        sum1 = np.dot(inputsData, self.weights) + self.biases
        return sum1
    
    def layer_backward(self, preWeights_Value, aftWetights_Demands):
        """
        执行层的反向传播

        参数:
        preWeights_Value: 前一层的权重值，形状为 (batch_size, n_inputs)
        aftWetights_Demands: 后一层的权重需求，形状为 (batch_size, n_neurons)

        返回:
        tuple: (norm_preActs_demands, norm_weights_adjust_matrixe)
            norm_preActs_demands: 归一化后的前一层激活前的需求，形状为 (batch_size, n_inputs)
            norm_weights_adjust_matrixe: 归一化后的权重调整矩阵，形状为 (n_inputs, n_neurons)
        """
        # 计算前一层的权重需求
        preWeights_demands = np.dot(aftWetights_Demands, self.weights.T)
        
        # 计算当前层的ReLU激活函数的导数
        condition = preWeights_Value > 0
        value_derivative = np.where(condition, 1, 0)
        
        # 计算前一层激活前的需求preActs_demands = preWeights_demands * value_derivative：
        # 这一步将上一步计算的误差与激活函数的导数相乘。
        # 这也是链式法则的应用，因为它考虑了激活函数对误差传播的影响。
        preActs_demands = preWeights_demands * value_derivative
        
        # 归一化前一层激活前的需求
        norm_preActs_demands = normalize(preActs_demands)

        # 计算权重调整矩阵
        weights_adjust_matrix = self.get_weights_adjust_matrix(preWeights_Value, aftWetights_Demands)
        
        # 归一化权重调整矩阵
        norm_weights_adjust_matrixe = normalize(weights_adjust_matrix)
        
        return (norm_preActs_demands,norm_weights_adjust_matrixe)
    
    def get_weights_adjust_matrix(self, preWeights_Value, aftWetights_Demands):
        # 创建与权重矩阵相同形状的全1矩阵
        plain_weights = np.full(self.weights.shape, 1.0)
        
        # 初始化权重调整矩阵
        weights_Adjust_Matrix = np.full(self.weights.shape, 0.0)
        
        # 转置全1矩阵
        plain_weights_T = plain_weights.T
        
        # 计算权重调整矩阵
        for i in range(BATCH_SIZE):
            weights_Adjust_Matrix += (plain_weights_T * preWeights_Value[i,:]).T * aftWetights_Demands[i,:]
        
        # 对权重调整矩阵取平均
        weights_Adjust_Matrix = weights_Adjust_Matrix / BATCH_SIZE
        return weights_Adjust_Matrix
class NetWork():
    def __init__(self, network_shape):
        self.layers = []
        self.network_shape = network_shape
        for i in range(len(self.network_shape) - 1):
            layer = Layer(network_shape[i], network_shape[i+1], i+1)
            self.layers.append(layer)
    def network_forward(self, inputsData):
        outputs = [inputsData]
        print(f"输入层数据:\n{inputsData}")
        
        for layer in self.layers:
            layer_sum = layer.layer_forward(outputs[-1])
            if layer.layer_number < len(self.layers):
                layer_output = activation_ReLU(layer_sum)
            else:
                layer_output = activation_softmax(layer_sum)
            outputs.append(layer_output)
            print(f"第{layer.layer_number}层输出结果:\n{layer_output}")
            print("...........................")
        
        return outputs
NETWORK_SHAPE = [2, 3, 4, 5, 2]
BATCH_SIZE = 10
def main():
    data = cp.creat_data(BATCH_SIZE)
    cp.plot_data(data,"原始")
    inputs = data[:,(0,1)]
    targets = np.copy(data[:,2])
    network = NetWork(NETWORK_SHAPE)
    outputs = network.network_forward(inputs)
    classifydata = classify(outputs[-1])
    
    print(f"分类结果:\n{classifydata}")
    print(f"原始标签:\n{data[:,2]}")
    precise_loss = precise_loss_func(outputs[-1], data[:,2])
    print(f"损失(接近1表示预测错误):\n{precise_loss}")
    cp.plot_data(data,"训练前")


    # 调整最后一层的目标值
    # 这行代码的意思是对神经网络最后一层的输出进行调整，以计算更精确的目标值。
    # 这个过程通常用于优化网络的训练，使预测结果更接近实际目标。
    # 具体的调整方法在 adjust_final_layer_targets 函数中实现。
    target_values = adjust_final_layer_targets(outputs[-1], targets)
    print(f"需求函数调整后的目标值:\n{target_values}")
    
    # 计算最后一层的权重调整矩阵
    adjust_matrix = network.layers[-1].get_weights_adjust_matrix(outputs[-2], target_values)
    print(f"权重矩阵的调整矩阵 △w:\n{adjust_matrix}")
    
    # 执行最后一层的反向传播
    layer_backward_result = network.layers[-1].layer_backward(outputs[-2], target_values)
    norm_preActs_demands, norm_weights_gradient, = layer_backward_result
    
    print(f"传递给第{network.layers[-2].layer_number}层的归一化误差 (形状: {norm_preActs_demands.shape}, 其中{norm_preActs_demands.shape[0]}是批量大小，{norm_preActs_demands.shape[1]}是该层神经元数量):\n{norm_preActs_demands}")
    print(f"第{network.layers[-1].layer_number}层的权重梯度矩阵 (形状: {norm_weights_gradient.shape}, 其中{norm_weights_gradient.shape[0]}是上一层神经元数量，{norm_weights_gradient.shape[1]}是当前层神经元数量):\n{norm_weights_gradient}")

if __name__ == '__main__':
    main()
