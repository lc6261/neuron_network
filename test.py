import numpy as np



# 损失函数
def precise_loss_func(predicted, real):
    """
    计算预测值和实际值之间的精确损失。
    
    参数:
    predicted: 预测的概率分布，形状为 (n_samples, n_classes)
    real: 实际的类别标签，形状为 (n_samples,)
    
    返回:
    每个样本的损失值，形状为 (n_samples,)
    """
    print(f"predicted:\n{predicted}")
    # 创建一个形状为 (len(real), 2) 的零矩阵
    real_matrix = np.zeros((len(real), 2))
    
    # 将实际值填充到矩阵的第二列
    real_matrix[:,1] = real
    
    # 将 1 减去实际值填充到矩阵的第一列
    real_matrix[:,0] = 1 - real
    
    print(f"real_matrix-2:\n{real_matrix}")
    
    # 计算预测值和实际值的点积，得到每个样本的损失
    product = np.sum(predicted * real_matrix, axis=1)
    
    # 返回每个样本的损失值
    return 1-product

a11 = -0.9
a12 = 0.4

a21 = 0.8
a22 = 0.5

a31 = 0.5
a32 = -0.8

a41 = -0.6
a42 = -0.1

a51 = 0.2
a52 = 0.8

inputs = np.array([
    [a11, a12],
    [a21, a22],
    [a31, a32],
    [a41, a42],
    [a51, a52]
])
real = np.array([0, 1, 0, 1, 0])


#print(precise_loss_func(inputs,real))



"""
逻辑解释
目标值矩阵的构建:
target 矩阵的第一列存储的是 1 - target_vector，即如果实际目标是0，则第一列为1；如果实际目标是1，则第一列为0。
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
    """
    print(f"predicted_values:\n{predicted_values}")
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



adjust = adjust_final_layer_targets(inputs,real)
print(f"adjust:\n{adjust}")

"""
PS H:\work\5.AI\pythonAI> & C:/Users/lc626/AppData/Local/Programs/Python/Python312/python.exe h:/work/5.AI/pythonAI/test.py
predicted_values:
[[-0.9  0.4]
 [-0.8  0.5]
 [ 0.5 -0.8]
 [-0.6 -0.1]
 [ 0.2  0.8]]
target_vector:
[0 1 0 1 0]
adjust:
[[ 1. -1.]
 [-1.  1.]
 [ 1. -1.]
 [-1.  1.]
 [ 1. -1.]]
# 分析结果:

# 1. predicted_values:
#    这是神经网络的预测输出。每行代表一个样本,每行有两个值,分别对应两个类别的预测概率。
#    例如,第一行 [-0.9  0.4] 表示第一个样本被预测为第一类的概率较低,第二类的概率较高。
# 这里概率较低是因为-0.9是一个负值,通常在神经网络输出中,较大的正值表示较高的概率,而负值或较小的值表示较低的概率。
# 相比之下,0.4虽然不是很大,但比-0.9大得多,所以第二类的概率相对较高。

# 2. target_vector:
#    这是真实的目标值。0表示第一类,1表示第二类。
#    例如,[0 1 0 1 0] 表示5个样本中,第2和第4个样本属于第二类,其余属于第一类。

# 3. adjust:
#    这是adjust_final_layer_targets函数的输出,表示需要对预测值进行的调整。
#    - [ 1. -1.] 表示需要增加第一个输出,减少第二个输出
#    - [-1.  1.] 表示需要减少第一个输出,增加第二个输出
   
#    具体分析:
#    - 第1个样本: 预测 [-0.9  0.4], 目标 0, 调整 [ 1. -1.] (正确,需要强化)
#    - 第2个样本: 预测 [-0.8  0.5], 目标 1, 调整 [-1.  1.] (错误,需要修正)
#    - 第3个样本: 预测 [ 0.5 -0.8], 目标 0, 调整 [ 1. -1.] (错误,需要修正)
#    - 第4个样本: 预测 [-0.6 -0.1], 目标 1, 调整 [-1.  1.] (错误,需要修正)
#    - 第5个样本: 预测 [ 0.5  0.3], 目标 0, 调整 [ 1. -1.] (错误,需要修正)

# 总结: 函数正确识别了预测错误的样本,并给出了适当的调整方向。
# 这些调整值将用于后续的反向传播过程,以更新神经网络的权重。

 """