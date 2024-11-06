# 文件名: quadratic_function_visualization.py
"""
1. 定义了一个二次函数 f(x, y) = x^2 + y^2。
2. 使用 NumPy 生成了用于绘图的数据点。
3. 创建了一个包含两个子图的图形:
   - 左侧是函数的3D表面图。
   - 右侧是函数的等高线图。
4. 在等高线图上,在点(1, 1)处绘制了偏导数的方向。
5. 最后调整了布局并显示图形。

这个可视化有助于理解二次函数在二维平面上的形状和性质,特别是通过等高线图和偏导数方向,可以直观地看到函数在不同点的变化趋势。
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义二次函数
def f(x, y):
    return x**2 + y**2

# 生成数据点
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 创建图形对象
fig = plt.figure(figsize=(12, 6))

# 绘制3D图
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax1.set_title('3D Surface Plot of $f(x, y) = x^2 + y^2$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)

# 绘制等高线图
ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z, 20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_title('Contour Plot of $f(x, y) = x^2 + y^2$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# 在点 (1, 1) 处绘制偏导数方向
x0, y0 = 1, 1
df_dx = 2 * x0  # x 方向的偏导数
df_dy = 2 * y0  # y 方向的偏导数
ax2.quiver(x0, y0, df_dx, df_dy, color='red', scale=10, label='Partial Derivatives at (1, 1)')
ax2.legend()

# 调整布局并显示图形
plt.tight_layout()
plt.show()
