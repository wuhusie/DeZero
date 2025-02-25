import numpy as np
from typing import Callable, Any
from step.step01 import Variable  # 使用相对导入

def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> float:
    """
    计算函数 f 在点 x 处的数值微分（数值梯度）。

    使用中心差分法（central difference）近似计算导数，通过微小扰动 eps 计算函数值的变化。

    Args:
        f (Callable[[Variable], Variable]): 输入一个 Variable 实例作为参数并返回一个 Variable 实例的函数。
            例如，f(x) = x ** 2 或其他自定义函数。
        x (Variable): 输入的 Variable 实例，表示计算微分的点，包含数据（如标量或张量）。
        eps (float, optional): 微小扰动值，用于计算数值微分。默认值为 1e-4（0.0001）。

    Returns:
        float: 函数 f 在点 x 处的数值微分（导数值）。结果是标量，表示 f(x) 在 x 处的变化率。

    Notes:
        - 该函数使用中心差分公式：(f(x + eps) - f(x - eps)) / (2 * eps)，以提高精度。
        - 假设输入的 Variable.data 是标量。如果处理张量，可能需要扩展为逐元素计算。
        - eps 值过小可能导致数值误差（舍入误差），过大可能降低精度，1e-4 通常是合理的默认值。
    """
    # 创建两个扰动后的 Variable 实例
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    
    # 计算函数在扰动点的值
    y0 = f(x0)
    y1 = f(x1)
    
    # 将 memoryview 转换为 numpy array 进行计算
    return float((np.array(y1.data) - np.array(y0.data)) / (2 * eps))
