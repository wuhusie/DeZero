import numpy as np
from typing import Callable, Any
from step.step01 import Variable  # 使用相对导入

def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4) -> float:
    """
    计算函数 f 在点 x 处的数值导数，用于近似导数计算。采用中心差分法，通过微小扰动 eps 评估函数变化，适用于验证解析导数或分析函数行为。

    Args:
        f (Callable[[Variable], Variable]): 输入一个接受 Variable 实例并返回 Variable 实例的可调用函数，要求函数在 x 附近连续且可微。例如，f(x) = x ** 2 或其他自定义函数。
        x (Variable): 输入的 Variable 实例，表示计算导数的点，包含数据（如标量或张量）。
        eps (float, optional): 微小扰动值，用于计算数值导数。默认值为 1e-4（0.0001），适用于大多数场景；对于高精度需求可减小至 1e-6，但需注意数值稳定性。

    Returns:
        float: 函数 f 在点 x 处的数值导数。结果是标量，表示 f(x) 在 x 处的变化率，仅适用于输入数据为标量的情况。

    Notes:
        - 该函数使用中心差分公式：(f(x + eps) - f(x - eps)) / (2 * eps)，以提高精度，相较前向差分误差更低。
        - 假设输入的 Variable.data 是标量。若需处理张量，需逐元素调用或扩展函数逻辑。
        - eps 值过小（如 1e-10）可能因浮点舍入误差导致结果不可靠，过大（如 0.1）则可能偏离真实导数。推荐范围为 1e-6 至 1e-3，默认为 1e-4。
        - 对于高曲率函数或噪声数据，数值导数可能不稳定，建议结合解析方法验证。

    Examples:
        >>> x = Variable(2.0)
        >>> f = lambda x: x ** 2
        >>> numerical_diff(f, x)
        4.000000000000051  # 近似 f'(x) = 2x 在 x = 2 处的值
    """
    # 创建两个扰动后的 Variable 实例
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    
    # 计算函数在扰动点的值
    y0 = f(x0)
    y1 = f(x1)
    
    # 将 memoryview 转换为 numpy array 进行计算
    return float((np.array(y1.data) - np.array(y0.data)) / (2 * eps))
