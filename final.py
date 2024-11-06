import numpy as np
import importlib
import matplotlib.pyplot as plt
cp = importlib.import_module("06_createdata_plot")

def adjust_final_layer_targets(predicted_values, target_vector):
    #print(f"target_vector:\n{target_vector}")
    
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
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:,1] = real
    real_matrix[:,0] = 1 - real
    
    product = np.sum(predicted * real_matrix, axis=1)
    
    # 使用 np.clip 来避免对数中的零值
    return -np.log(np.clip(product, 1e-15, 1.0))

def activation_ReLU(input):
    return np.maximum(0, input)

def activation_softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

def classify(input_probabilities):
    return np.rint(input_probabilities[:,1])

def activation_LeakyReLU(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def activation_LeakyReLU_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

NETWORK_SHAPE = [2, 32, 16, 8, 2]  # 简化网络结构

class Layer():
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        # 使用 Xavier 初始化
        self.weights = np.random.randn(self.n_inputs, self.n_neurons) * np.sqrt(1.0 / n_inputs)
        self.biases = np.zeros(self.n_neurons)
        self.last_input = None

    def layer_forward(self, inputsData):
        self.last_input = inputsData
        sum1 = np.dot(inputsData, self.weights) + self.biases
        return sum1

    def backward(self, output_error, learning_rate, l2_lambda=0.01):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.last_input.T, output_error)
        
        # 添加 L2 正则化
        weights_error += l2_lambda * self.weights
        
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.mean(output_error, axis=0)
        
        return input_error

class NetWork():
    def __init__(self, network_shape):
        self.layers = []
        self.network_shape = network_shape

        for i in range(len(self.network_shape) - 1):
            layer = Layer(network_shape[i], network_shape[i+1])
            self.layers.append(layer)

    def network_forward(self, inputsData):
        outputs = [inputsData]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[-1])
            if i < (len(self.layers)-1):
                layer_output = activation_LeakyReLU(layer_sum)
            else:
                layer_output = activation_softmax(layer_sum)
            outputs.append(layer_output)
        return outputs

    def backward(self, error, learning_rate):
        for i in reversed(range(len(self.layers))):
            if i != len(self.layers) - 1:
                # 使用当前层的输出（即下一层的输入）来计算导数
                error = error * activation_LeakyReLU_derivative(self.layers[i+1].last_input)
            error = self.layers[i].backward(error, learning_rate)

def train(network, inputs, targets, epochs, initial_learning_rate):
    learning_rate = initial_learning_rate
    for epoch in range(epochs):
        outputs = network.network_forward(inputs)
        loss = cross_entropy_loss(outputs[-1], np.column_stack((1-targets, targets)))
        
        if np.isnan(loss):
            print(f"警告：第 {epoch} 轮出现 NaN 损失")
            break
        
        error = outputs[-1] - np.column_stack((1-targets, targets))
        
        # 梯度裁剪
        error = np.clip(error, -1, 1)
        
        network.backward(error, learning_rate)
        
        if epoch % 100 == 0:
            print(f"第 {epoch} 轮，损失：{loss}")
        
        # 学习率衰减
        learning_rate *= 0.9999

def cross_entropy_loss(predicted, real):
    epsilon = 1e-15
    predicted = np.clip(predicted, epsilon, 1 - epsilon)
    N = predicted.shape[0]
    ce_loss = -np.sum(real * np.log(predicted) + (1 - real) * np.log(1 - predicted)) / N
    return ce_loss

def main():
    # 设置超参数
    num_data_points = 1000
    epochs = 10000  # 增加训练轮数
    learning_rate = 0.0005  # 调整初始学习率

    # 创建数据
    data = cp.creat_data(num_data_points)
    
    # 绘制原始数据
    cp.plot_data(data, "原始数据")
    
    inputs = data[:, (0, 1)]
    targets = np.copy(data[:, 2])

    # 创建并训练网络
    network = NetWork(NETWORK_SHAPE)
    train(network, inputs, targets, epochs=epochs, initial_learning_rate=learning_rate)

    # 训练后进行预测
    outputs = network.network_forward(inputs)
    classifydata = classify(outputs[-1])
    
    # 打印结果
    print(f"训练后分类结果:\n{classifydata}")
    print(f"原始标签:\n{data[:, 2]}")
    precise_loss = precise_loss_func(outputs[-1], data[:, 2])
    print(f"训练后损失:\n{precise_loss}")

    # 更新数据集的分类结果
    data[:, 2] = classifydata

    # 绘制训练后的数据
    cp.plot_data(data, f"训练后数据 (epochs={epochs})")

    # 确保显示图形
    plt.show()

if __name__ == '__main__':
    main()
