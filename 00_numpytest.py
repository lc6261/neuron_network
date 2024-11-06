"""
0.numpy学习
"""
import numpy as np

#1.生成array
print("1.生成array")
array1 = np.zeros((2,3))#创建2行3列的数组，数值为0
array2 = np.full((2,3),3)#创建2行3列的数组，填充数值为3

print(array1)
print(array2)


#2.创建矩阵，填充矩阵
print("2.创建矩阵，填充矩阵")
list1 = [[0,1,2],
         [3,4,5]]#创建2行3列的列表

array3 = np.array(list1)#转为2行3列数组
print(list1)
print(array3)

a = array3[0,:]#提取数组的第0行的所有列
print(a)

b = array3[:,2]#提取数组的所有行的第2列
print(b)



#3.点乘(点积),矩阵乘法要求第一个矩阵的列数必须等于第二个矩阵的行数，并且结果矩阵的形状是第一个矩阵的行数乘以第二个矩阵的列数

print("3.点乘(点积):(2,3)dot(3,2)")
list3 = [[0,1,2],
         [3,4,5]]#创建2行3列的列表
array3 = np.array(list3)#转为2行3列数组


list4 = [[0,1],
         [2,3],
         [4,5]]#创建3行2列的列表
array4 = np.array(list4)#转为3行2列数组


array5 = np.dot(array3,array4)
#             [0,1]
#             [2,3]
#       dot   [4,5]   =                               = 
#[a,b,c]                 [a*0+b*2+c*4,  a*1+b*3+c*5]       [0+2+8,  0+3+10]
#  
#[d,e,f]                 [d*0+e*2+f*4,  d*1+e*3+f*5]       [0+8+20,  3+12+25]
print(array5)
print("3.点乘(点积):(3,2)dot(2,3)")

array6 = np.dot(array4,array3)
#             [a,b,c]  
#           
#       dot   [d,e,f]  =                              = 
#[0,1]                    [0*a+1*d,  0*b+1*e,  0*c+1*f]       [0*0+1*3,  0*1+1*4,  0*2+1*5]
#[2,3]                    [2*a+3*d,  2*b+3*e,  2*c+3*f]       [2*0+3*3,  2*1+3*4,  2*2+3*5]
#[4,5]                    [4*a+5*d,  4*b+5*e,  4*c+5*f]       [4*0+5*3,  4*1+5*4,  4*2+5*5]
print(array6)


print("3.点乘(点积):2")
list7 = [[0,1],
         [2,3]]#创建3行2列的列表
array7 = np.array(list7)#转为3行2列数组
array8 = np.dot(array4,array7)
#             [a,b]  
#           
#       dot   [d,e]  =                         = 
#[0,1]                    [0*a+1*d,  0*b+1*e]     [0*0+1*3,  0*1+1*4]
#[2,3]                    [2*a+3*d,  2*b+3*e]     [2*0+3*3,  2*1+3*4]
#[4,5]                    [4*a+5*d,  4*b+5*e]     [4*0+5*3,  4*1+5*4]
print(array8)



print("4.花乘,相加:* + 矩阵大小必须相同")

list8 = [[0,1],
         [2,3]]#创建3行2列的列表
array8 = np.array(list8)#转为3行2列数组

list9 = [[0,1],
         [2,3]]#创建3行2列的列表
array9 = np.array(list9)#转为3行2列数组


array10 = array8 * array9
#           [a,b]  
#       *   [c,d]  = 
#[0,1]                    [0*a,  1*b] 
#[2,3]                    [2*c,  3*d]
print(array10)

array11 = array8 + array9
#           [a,b]  
#       +   [c,d]  = 
#[0,1]                    [0+a,  1+b] 
#[2,3]                    [2+c,  3+d]
print(array11)

print("[0,1] + [[0,1],[2,3]]=")
array9 = np.array([0,1])
array11 = array8 + array9
#           [a,b]                      
#       +   [c,d]  =                   = 
#[0,1]                    [0+a,  1+b]         [0+0,  1+1] 
#                         [0+c,  1+d]         [0+2,  1+3]
print(array11)

print("5.矩阵 取最大值,求最大值")

array12 = np.array([[-1,3],
                   [2,-2],
                   [-1,-3]])#多加一个[]变成list，然后在用array转换为矩阵

print(np.maximum(0,array12))#2个矩阵各点 相对于0求最大值


array13 = np.array([[3,3],
                   [2,-2],
                   [2,-3]])#多加一个[]变成list，然后在用array转换为矩阵

print(np.maximum(array12,array13))#2个矩阵各点求最大

print(np.max(array12,axis = 1))#axis = 0 当前矩阵所有列求最大值,axis = 1 当前矩阵所有行求最大值
 

arr = np.array([1, 2, 3, 4, 5])
indices = np.where(arr > 2)# 输出: (array([2, 3, 4]),)  # 索引 2, 3, 4 处的元素大于 2
print(indices)


a = np.array([10, 20, 30, 40, 50])
b = np.array([100, 200, 300, 400, 500])
result = np.where(a > 25, a, b)
# 输出: array([100, 200,  30,  40,  50])  # a 中大于 25 的元素保留，否则取 b 中的元素
print(result)


matrix = np.array([[1, 2], [3, 4]])
result = np.where(matrix > 2, 0, 1)
# 输出: array([[1, 1],
#              [0, 0]])  # 大于 2 的元素变为 0，其余变为 1
print(result)