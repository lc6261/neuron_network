import numpy as np
import importlib
import copy
import math

cp = importlib.import_module("06_createdata_plot")

def normalize(input_array):
    max_values = np.max(np.absolute(input_array),axis=1,keepdims=True)
    scaled_rate = np.where(max_values == 0, 1, 1 / max_values)
    norm_data = input_array * scaled_rate
    return norm_data
    
def vector_normalize(input_array):
    max_values = np.max(np.absolute(input_array))
    scaled_rate = np.where(max_values == 0, 1, 1 / max_values)
    norm_data = input_array * scaled_rate
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

def loss_function(predicted, real):
    # 将预测值二值化：大于0.5的变为1，小于等于0.5的变为0
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition, 1, 0)
    
    # 创建真实值的矩阵表示
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real  # 第二列为原始真实值
    real_matrix[:, 0] = 1 - real  # 第一列为1减去真实值（即相反的类别）
    
    # 计算二值化预测值和真实值矩阵的点积
    product = np.sum(binary_predicted * real_matrix, axis=1)

    # 返回1减去点积的结果作为损失
    return 1 - product

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
    def network_backward(self, layer_outputs, targets_vector):
        backup_network = copy.deepcopy(self)        #备用网络
        print(f"backup_network:\n{backup_network}")

        preActs_demands = adjust_final_layer_targets(layer_outputs[-1], targets_vector)

        for layer_index in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers)-(layer_index+1)] #倒序
            if layer_index!=0:
                layer.biases -= LEARNING_RATE * np.mean(preActs_demands, axis=0)
                layer.biases = vector_normalize(layer.biases)
            
            outputs = layer_outputs[len(layer_outputs)-(layer_index+2)]
            result_list = layer.layer_backward(outputs, preActs_demands)
            preActs_demands = result_list[0]

            weight_adjust_matrix = result_list[1]
            layer.weights += LEARNING_RATE * weight_adjust_matrix
            layer.weights = normalize(layer.weights)
        return backup_network
    #单批次处理
    def one_batch_train(self, batch):
        inputs = batch[:, (0, 1)]
        targets = copy.deepcopy(batch[:, 2]).astype(int)  # 标准答案
        outputs = self.network_forward(inputs)
        precise_loss = precise_loss_func(outputs[-1], targets)

        if np.mean(precise_loss) <= 0.1:
            print('无需训练')
        else:
            backup_network = self.network_backward(outputs, targets)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = precise_loss_func(backup_outputs[-1], targets)

            if np.mean(precise_loss) >= np.mean(backup_precise_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print('已改进')
            else:
                print('未改进')
        print(f"-------------------------------------------")

    def train(self, n_entries):
        n_batches = math.ceil(n_entries/BATCH_SIZE)
        for i in range(n_batches):
            batch = cp.creat_data(BATCH_SIZE)
            self.one_batch_train(batch)
        
        data = cp.creat_data(100)
        cp.plot_data(data, "正确分类")
        inputs = data[:, (0, 1)]
        outputs = self.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "训练后")

LEARNING_RATE = 0.01#学习率 = 每次调整 1%
NETWORK_SHAPE = [2, 3, 4, 5, 2]
BATCH_SIZE = 10

def main():

    # 创建网络实例
    network = NetWork(NETWORK_SHAPE)
    '''
    
    # 创建初始数据
    data = cp.creat_data(BATCH_SIZE)
    
    # 获取输入数据（前两列）
    inputs = data[:, (0, 1)]
    
    # 通过网络前向传播
    outputs = network.network_forward(inputs)
    
    # 对输出进行分类
    classification = classify(outputs[-1])
    
    # 更新数据的分类结果
    data[:, 2] = classification
    
    # 绘制训练前的数据
    cp.plot_data(data, "Before training")
    '''
    # 获取用户输入的训练数据量
    n_entries = int(input("请输入用于训练的数据条目数: \n"))
    
    # 使用指定数量的数据进行训练
    network.train(n_entries)

if __name__ == '__main__':
    main()
