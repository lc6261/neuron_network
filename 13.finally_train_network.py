import numpy as np
import importlib
import copy
import math

cp = importlib.import_module("06_createdata_plot")

NETWORK_SHAPE = [2, 100, 200, 100, 50, 2]
BATCH_SIZE = 30
LEARNING_RATE = 0.015
LOSS_THRESHOLD = 0.1
FORCE_TRAIN_THRESHOLD = 0.05

force_train = False
random_train = False
n_improved = 0
n_not_improved = 0
current_loss = 1


#标准化函数
def normalize(array):
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array * scale_rate
    return norm

#向量标准化函数
def vector_normalize(array):
    max_number = np.max(np.absolute(array))
    scale_rate = np.where(max_number == 0, 1, 1/max_number)
    norm = array * scale_rate
    return norm

# 激活函数
def activation_ReLU(inputs):
    return np.maximum(0, inputs)

#分类函数
def classify(probabilities):
    classification = np.rint(probabilities[:, 1])
    return classification
    
# softmax激活函数
def activation_softmax(inputs):
    max_values = np.max(inputs, axis=1, keepdims=True)
    slided_inputs = inputs - max_values
    exp_values = np.exp(slided_inputs)
    norm_base = np.sum(exp_values, axis=1, keepdims=True)
    norm_values = exp_values/norm_base
    return norm_values

#损失函数1
def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(predicted*real_matrix, axis=1)
    return 1 - product

#损失函数2
def loss_function(predicted, real):
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition, 1, 0)
    real_matrix = np.zeros((len(real), 2))
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real
    product = np.sum(binary_predicted*real_matrix, axis=1)
    return 1 - product

#需求函数，是当前层的误差信号或梯度。
def get_final_layer_preAct_damands(predicted_values, target_vector):
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector
    
    for i in range(len(target_vector)):
        if np.dot(target[i], predicted_values[i]) > 0.5:
            target[i] = np.array([0, 0])
        else:
            target[i] = (target[i] - 0.5) * 2
    return target

#-------------------------------------------------------------
#定义一个层类
class Layer: 
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(n_neurons)
    
    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    '''
    输入:
    +------------------------+      +------------------------+
    |   preWeights_values    |      |  afterWeights_demands  |
    | (前一层的输出, 即x)     |      | (后一层的梯度, 即dL/dy) |
    +------------------------+      +------------------------+
            |                               |
            |                               |
            v                               v
    +-------------------------------------------------------+
    |                     计算过程                           |
    |                                                       |
    |  1. preWeights_damands = np.dot(afterWeights_demands, |
    |                                 self.weights.T)       |
    |     (计算当前层的预激活梯度 dL/dz)                      |
    |                                                       |
    |  2. condition = (preWeights_values > 0)               |
    |     value_derivatives = np.where(condition, 1, 0)     |
    |     (计算ReLU的导数)                                   |
    |                                                       |
    |  3. preActs_demands = value_derivatives *             |
    |                       preWeights_damands              |
    |     (计算前一层的梯度 dL/dx)                            |
    |                                                       |
    |  4. norm_preActs_demands = normalize(preActs_demands) |
    |     (标准化前一层的梯度)                                |
    |                                                       |
    |  5. weight_adjust_matrix = get_weight_adjust_matrix() |
    |     +---------------------------------------------+   |
    |     | get_weight_adjust_matrix 详细过程:           |   |
    |     | a. 创建与权重相同形状的全1矩阵和全0矩阵        |   |
    |     | b. 对每个批次样本:                           |   |
    |     |    weights_adjust_matrix +=                 |   |
    |     |      (plain_weights_T * preWeights_values[i]).T|   |
    |     |      * aftWeights_demands[i]                |   |
    |     | c. 取平均: weights_adjust_matrix /= BATCH_SIZE|   |
    |     +---------------------------------------------+   |
    |                                                       |
    |  6. norm_weight_adjust_matrix =                       |
    |       normalize(weight_adjust_matrix)                 |
    |     (标准化权重调整矩阵)                                |
    +-------------------------------------------------------+
            |                               |
            |                               |
            v                               v
    +------------------------+      +------------------------+
    | norm_preActs_demands   |      |norm_weight_adjust_matrix|
    | (标准化的dL/dx)         |      | (标准化的dL/dW)         |
    +------------------------+      +------------------------+
    输出
    '''
    def layer_backward(self, preWeights_values, afterWeights_demands):
        preWeights_damands = np.dot(afterWeights_demands, self.weights.T)
        
        condition = (preWeights_values > 0)
        value_derivatives = np.where(condition, 1, 0)
        
        preActs_demands = value_derivatives*preWeights_damands
        norm_preActs_demands = normalize(preActs_demands)
        
        weight_adjust_matrix = self.get_weight_adjust_matrix(preWeights_values, afterWeights_demands)
        norm_weight_adjust_matrix = normalize(weight_adjust_matrix)
        
        return (norm_preActs_demands, norm_weight_adjust_matrix)
    
    
    """
    好的,我来为您解释这个get_weight_adjust_matrix方法的功能,并用图解说明。
    这个方法的主要目的是计算权重调整矩阵,用于更新神经网络层的权重。让我用中文解释一下过程:
    创建初始矩阵:
    plain_weights: 与权重相同形状的全1矩阵
    weights_adjust_matrix: 与权重相同形状的全0矩阵
    对每个批次样本进行循环计算:
    对每个样本,计算 (plain_weights_T * preWeights_values[i]).T * aftWeights_demands[i]
    将结果累加到 weights_adjust_matrix
    最后,将累加结果除以批次大小(BATCH_SIZE)得到平均值
    这里是一个简化的图解:
    这个方法实际上是在计算权重梯度的估计值。
    它利用了输入值(preWeights_values)和输出梯度(aftWeights_demands)来计算权重应该如何调整。
    这是反向传播算法的核心部分,用于指导神经网络如何更新其权重以减小误差。
    """
    # 用了输入值(preWeights_values)和输出梯度(aftWeights_demands)来计算权重应该如何调整。
    # 这是反向传播算法的核心部分,用于指导神经网络如何更新其权重以减小误差。
    def get_weight_adjust_matrix(self, preWeights_values, aftWeights_demands):
        #创建与权重相同形状的全1矩阵 (plain_weights) 和全0矩阵 (weights_adjust_matrix)。
        plain_weights = np.full(self.weights.shape, 1)
        weights_adjust_matrix = np.full(self.weights.shape, 0.0)

        #一次处理一批样本,并对本批次的样本进行循环处理,然后计算权重调整矩阵平均值
        #计算 (plain_weights_T * preWeights_values[i]).T * aftWeights_demands[i]
        #将结果累加到 weights_adjust_matrix
        plain_weights_T = plain_weights.T
        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T*preWeights_values[i, :]).T * aftWeights_demands[i, :]
        
        #最后,将累加结果除以批次大小(BATCH_SIZE)得到平均值
        weights_adjust_matrix = weights_adjust_matrix/BATCH_SIZE
        
        #这个最终的weights_adjust_matrix就是用来调整权重的矩阵。 它反映了在这个小批量中,每个权重应该如何调整以减小误差。
        return weights_adjust_matrix
        
#-----------------------------------------------------------------   
#定义一个网络类
class Network:
    def __init__(self, network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape)-1):
            layer = Layer(network_shape[i], network_shape[i+1])
            self.layers.append(layer)
     
    #前馈运算函数 
    def network_forward(self, inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers)-1:
                layer_output = activation_ReLU(layer_sum)
                layer_output = normalize(layer_output)
            else:
                layer_output = activation_softmax(layer_sum)
            outputs.append(layer_output)
        return outputs
    
    #反向传播函数,入参是前馈运算的输出layer_outputs和目标向量target_vector
    def network_backward(self, layer_outputs, target_vector):
        backup_network = copy.deepcopy(self) # 备用网络
        # 使用 get_final_layer_preAct_damands 计算输出层的误差信号。
        # 误差信号指示了网络输出需要如何调整以改进预测。
        # a.指导学习方向：误差信号指示了网络输出需要如何调整以改进预测。
        # b.启动反向传播：这个信号作为反向传播的起点，从输出层开始向前传播。
        # c.更新权重和偏置：虽然不是精确的梯度，但这个信号仍用于更新网络参数。在您的代码中，它被用来更新偏置。
        # d.简化的学习机制：这种方法可能在某些简单任务中有效，特别是在二分类问题中。
        # e.自适应学习：对于正确分类的样本，信号为零，这意味着网络不会继续调整已经正确的预测。
        preAct_demands = get_final_layer_preAct_damands(layer_outputs[-1], target_vector)
        for i in range(len(self.layers)):
            layer = backup_network.layers[len(self.layers) - (1+i)] # 倒序，选择当前层:
            if i != 0:#这个条件确保我们不在输入层更新偏置，因为输入层通常没有偏置。
                #LEARNING_RATE 是学习率，控制每次更新的步长。
                #np.mean(preAct_demands, axis=0) 计算批次中所有样本的平均梯度。
                layer.biases += LEARNING_RATE * np.mean(preAct_demands, axis=0)
                #标准化可以帮助控制偏置的大小，防止它们变得过大。
                layer.biases = vector_normalize(layer.biases)

            #获取前一层输出: preLayer_outputs = layer_outputs[len(layer_outputs) - (2+i)]
            preLayer_outputs = layer_outputs[len(layer_outputs) - (2+i)]

            # 前一层输出 preLayer_outputs = preWeights_values
            # 当前层的需求 preAct_demands = afterWeights_demands
            # 执行当前层的反向传播:
            results_list = layer.layer_backward(preLayer_outputs, preAct_demands)
            
            
            # 前一层需求 preAct_demands
            # 权重调整矩阵 weights_adjust_matrix    
            preAct_demands = results_list[0]
            weights_adjust_matrix = results_list[1]
            #更新权重
            layer.weights += LEARNING_RATE * weights_adjust_matrix
            layer.weights = normalize(layer.weights)
        return backup_network
    
    #单批次训练
    def one_batch_train(self, batch):
        global force_train, random_train, n_improved, n_not_improved

        inputs = batch[:,(0, 1)]
        targets = copy.deepcopy(batch[:, 2]).astype(int) # 标准答案
        outputs = self.network_forward(inputs)
        precise_loss = precise_loss_function(outputs[-1], targets)
        loss = loss_function(outputs[-1], targets)
            
        if np.mean(loss) <= LOSS_THRESHOLD:#损失函数小于这个值就不需要训练了
            print('No need for training')
        else:
            backup_network = self.network_backward(outputs, targets)
            backup_outputs = backup_network.network_forward(inputs)
            backup_precise_loss = precise_loss_function(backup_outputs[-1], targets)
            backup_loss = loss_function(backup_outputs[-1], targets)
            
            if np.mean(precise_loss) >= np.mean(backup_precise_loss) or np.mean(loss) >= np.mean(backup_loss):
                for i in range(len(self.layers)):
                    self.layers[i].weights = backup_network.layers[i].weights.copy()
                    self.layers[i].biases = backup_network.layers[i].biases.copy()
                print('Improved')
                n_improved += 1
            
            else:
                if force_train:
                    for i in range(len(self.layers)):
                        self.layers[i].weights = backup_network.layers[i].weights.copy()
                        self.layers[i].biases = backup_network.layers[i].biases.copy()
                    print('Force train')
                if random_train:
                    self.random_update()
                    print("Random update")
                else:  
                    print('No improvement')
                n_not_improved += 1
        print('-----------------------------------------')
            
    #多批次训练
    def train(self, n_entries):
        global force_train, random_train, n_improved, n_not_improved
        n_improved = 0
        n_not_improved = 0

        n_batches = math.ceil(n_entries/BATCH_SIZE)
        for i in range(n_batches):
            batch = cp.creat_data(BATCH_SIZE)
            self.one_batch_train(batch)
        
        # 添加检查，避免除以零的错误
        if n_improved + n_not_improved > 0:
            improvement_rate = n_improved / (n_improved + n_not_improved)
        else:
            improvement_rate = 0  # 或者其他适当的默认值

        print("改进率")
        print(format(improvement_rate, ".0%"))
        
        if improvement_rate <= FORCE_TRAIN_THRESHOLD:
            force_train = True
        else:
            force_train = False
        if n_improved == 0:
            random_train = True
        else:
            random_train = False
        
        data = cp.creat_data(800)
        inputs = data[:, (0, 1)]
        outputs = self.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "After training")
        
    #随机更新
    def random_update(self):
        random_network = Network(NETWORK_SHAPE)
        for i in range(len(self.layers)):
            weights_change = random_network.layers[i].weights
            biases_change = random_network.layers[i].biases
            self.layers[i].weights += weights_change
            self.layers[i].biases += biases_change
            
    #核验LOSS值
    def check_loss(self):
        data = cp.creat_data(1000)
        inputs = data[:, (0, 1)]
        targets = copy.deepcopy(data[:, 2]).astype(int) # 标准答案
        outputs = self.network_forward(inputs)
        loss = loss_function(outputs[-1], targets)
        return np.mean(loss)
        
#-------------MAIN-------------------------
def main():
    global current_loss
    data = cp.creat_data(800) #生成数据
    cp.plot_data(data, "Right classification")

    #选择起始网络
    use_this_network = 'n' #No
    while use_this_network != 'Y' and use_this_network != 'y':
        network = Network(NETWORK_SHAPE)
        inputs = data[:, (0, 1)]
        outputs = network.network_forward(inputs)
        classification = classify(outputs[-1])
        data[:, 2] = classification
        cp.plot_data(data, "Choose network")
        use_this_network = input("Use this network? Y to yes, N to No \n")
    
    #进行训练
    do_train = input("Train? Y to yes, N to No \n")
    while do_train == 'Y' or do_train == 'y' or do_train.isnumeric() == True:
        if do_train.isnumeric() == True:
            n_entries = int(do_train)
        else:
            n_entries = int(input("Enter the number of data entries used to train. \n"))
            
        network.train(n_entries)
        do_train = input("Train? Y to yes, N to No \n")
        
    #演示训练效果
    inputs = data[:, (0, 1)]
    outputs = network.network_forward(inputs)
    classification = classify(outputs[-1])
    data[:, 2] = classification
    cp.plot_data(data, "After training")
    print("谢谢，再见！")
#----------------TEST-------------------------
def test():
    pass

#--------------运行---------------------
main()
'''
+--------------------+      +--------------------+      +--------------------+
|     输入数据        |      |     隐藏层(多层)    |      |       输出层       |
| preWeights_values  |----->| 每层:               |----->| 最终输出            |
| (x)                |      | z = Wx + b         |      | softmax激活        |
| 形状:(batch_size,   |      | y = ReLU(z)        |      | 形状:(batch_size,  |
|      n_inputs)     |      | layer_forward()    |      |      n_outputs)    |
+--------------------+      +--------------------+      +--------------------+
         |                           |                           |
         |                           |                           |
         |                           |                           v
         |                           |             +----------------------------+
         |                           |             |          损失函数           |
         |                           |             | L = loss_function(预测, 真实)|
         |                           |             | precise_loss_function()    |
         |                           |             | loss_function()            |
         |                           |             +----------------------------+
         |                           |                           |
         |                           |                           |
         |                           |                           v
         |                           |             +----------------------------+
         |                           |             | get_final_layer_preAct_damands|
         |                           |             | 计算输出层的误差信号          |
         |                           |             | 输入: 预测值, 目标值          |
         |                           |             | 输出: 误差信号 (不是严格的导数)|
         |                           |             +----------------------------+
         |                           |                           |
         |                           |                           |
         v                           v                           v
+--------------------+      +--------------------+      +--------------------+
|   输入层反向传播     |      |   隐藏层反向传播    |      |   输出层误差信号    |
| dL/dx              |<-----| 每层:               |<-----| afterWeights_demands|
| 形状:(batch_size,   |      | dL/dz, dL/dW, dL/db|      | 形状:(batch_size,  |
|      n_inputs)     |      | layer_backward()   |      |      n_outputs)    |
+--------------------+      +--------------------+      +--------------------+
                                     |
                                     |
                                     v
              +--------------------------------------------------+
              |                  反向传播计算                      |
              | 1. dy/dz (ReLU导数)                               |
              |    value_derivatives                              |
              |    np.where(preWeights_values > 0, 1, 0)          |
              |    形状: (batch_size, n_neurons)                  |
              | 2. dL/dz = dL/dy * dy/dz                          |
              |    preWeights_damands                             |
              |    np.dot(afterWeights_demands, self.weights.T)   |
              |    形状: (batch_size, n_neurons)                  |
              | 3. dL/dx = dL/dz * dy/dz                          |
              |    preActs_demands                                |
              |    value_derivatives * preWeights_damands         |
              |    形状: (batch_size, n_inputs)                   |
              | 4. dL/dW = x^T * dL/dz                            |
              |    weights_adjust_matrix                          |
              |    get_weight_adjust_matrix()                     |
              |    形状: (n_inputs, n_neurons)                    |
              +--------------------------------------------------+
                                     |
                                     |
                                     v
              +--------------------------------------------------+
              |                    权重更新                       |
              | W += LEARNING_RATE * weights_adjust_matrix        |
              | 形状: W (n_inputs, n_neurons)                     |
              |      weights_adjust_matrix (同上)                 |
              |                                                  |
              | if i != 0: # 不更新输入层                          |
              |   b += LEARNING_RATE * np.mean(preActs_demands, axis=0)|
              |   b = vector_normalize(b)                        |
              | 形状: b (n_neurons,)                              |
              |      np.mean(preActs_demands, axis=0) (同上)      |
              |                                                  |
              | get_weight_adjust_matrix 方法:                    |
              | 1. 初始化 weights_adjust_matrix 为零              |
              | 2. 对每个批次样本:                                 |
              |    weights_adjust_matrix +=                      |
              |      (preWeights_values[:, :, np.newaxis] *      |
              |       preWeights_damands[:, np.newaxis, :])      |
              | 3. 取平均: weights_adjust_matrix /= batch_size    |
              +--------------------------------------------------+
              


为了用ASCII字符表示你的代码中的链式法则，我们可以使用简单的箭头和数学符号来表示。以下是Layer类的layer_backward方法中链式法则的ASCII表示：

preWeights_values (x)  ---> layer_sum (z)  ---> afterWeights_demands (y)
                            ReLU activation

dy/dx = (dy/dz) * (dz/dx)
(dy/dz) = afterWeights_demands
   (dz/dx) = value_derivatives = np.where(preWeights_values > 0, 1, 0)
preWeights_damands = np.dot(afterWeights_demands, self.weights.T)
   preActs_demands = value_derivatives * preWeights_damands
   norm_preActs_demands = normalize(preActs_demands)

   

最底下的注释是对的，它使用ASCII字符表示了Layer类的layer_backward方法中链式法则的计算过程。以下是详细解释：

preWeights_values (x) 表示输入值，也就是前一层的输出。
layer_sum (z) 表示当前层的加权和，也就是 np.dot(inputs, self.weights) + self.biases 的结果。
afterWeights_demands (y) 表示后一层的需求，也就是下一层的 preAct_demands。
ReLU activation 表示ReLU激活函数，它将 layer_sum (z) 转换为当前层的输出。
链式法则的计算过程如下：

dy/dx 表示后一层的需求对输入值的导数，也就是我们想要计算的 preActs_demands。
(dy/dz) 表示后一层的需求对加权和的导数，也就是 afterWeights_demands。
(dz/dx) 表示加权和对输入值的导数，也就是 value_derivatives，它可以通过ReLU激活函数的导数计算得到，即 np.where(preWeights_values > 0, 1, 0)。
因此，根据链式法则，我们可以计算 preActs_demands：

preWeights_damands = np.dot(afterWeights_demands, self.weights.T) 计算后一层的需求对当前层的加权和的导数。
preActs_demands = value_derivatives * preWeights_damands 计算后一层的需求对当前层的输入值的导数。
norm_preActs_demands = normalize(preActs_demands) 对计算结果进行标准化。
所以，最底下的注释正确地表示了Layer类的layer_backward方法中链式法则的计算过程。


"后一层的需求"是指在神经网络的反向传播过程中，当前层的输出（也就是后一层的输入）对损失函数的导数。
在Layer类的layer_backward方法中，"后一层的需求"指的是afterWeights_demands，
它表示下一层的preAct_demands，也就是下一层的输入对损失函数的导数。

在反向传播过程中，我们从输出层开始，逐层计算每个层的输入对损失函数的导数，
然后使用这些导数来更新每个层的权重和偏置。因此，"后一层的需求"实际上是当前层的输出对损失函数的导数，
它反映了当前层的输出对整个网络性能的影响，并用于指导当前层的权重和偏置的更新。
'''