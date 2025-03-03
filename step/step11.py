from __future__ import annotations
from typing import Optional, Any, Union, List
import numpy as np

class Variable:
    """
    表示计算图中的变量，封装数据并支持自动求导。

    该类用于存储计算过程中的数据、梯度以及创建者信息，通过 backward 方法实现反向传播。适用于构建动态计算图，例如神经网络的前向和反向计算。

    Attributes:
        data (np.ndarray): 变量的数据，通常为标量或张量。
        grad (Optional[np.ndarray]): 变量的梯度，初始为 None，在反向传播时计算。
        creator (Optional[Function]): 创建该变量的函数实例，初始为 None，用于构建计算图。

    Methods:
        __init__: 初始化变量，验证并存储输入数据。
        set_creator: 设置变量的创建者函数，建立计算图关系。
        backward: 执行反向传播，计算并传播梯度。

    Notes:
        - 输入数据必须是 np.ndarray 类型，否则抛出 TypeError。
        - 反向传播假设 creator 是 Function 实例，支持单输入单输出场景。
        - 若需支持多输入场景，需在 Function 和 backward 中扩展逻辑。
    """

    def __init__(self, data: Optional[np.ndarray]) -> None:
        """
        初始化 Variable 实例，验证并存储输入数据。

        Args:
            data (Optional[np.ndarray]): 输入数据，可以是 NumPy 数组或 None。若为 None，则创建空变量。

        Raises:
            TypeError: 如果 data 不是 np.ndarray 类型且不为 None，则抛出此异常。

        Notes:
            - 数据验证确保后续计算的兼容性。
            - grad 和 creator 初始为 None，在计算图构建和反向传播中动态设置。
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data).__name__} is not supported")
        self.data: Optional[np.ndarray] = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None

    def set_creator(self, func: Function) -> None:
    # def set_creator(self, func: Any) -> None:
        """
        设置变量的创建者函数，建立计算图关系。

        Args:
            func (Function): 创建该变量的 Function 实例，用于反向传播时的梯度计算。

        Notes:
            - 该方法通常由 Function.__call__ 调用，以记录计算图中的依赖关系。
        """
        self.creator = func

    def backward(self) -> None:
        """
        执行反向传播，计算并传播梯度至输入变量。

        通过调用创建者函数的 backward 方法，基于链式法则计算梯度，并递归传播至计算图中的前驱节点。

        Notes:
            - 若 grad 为 None，则初始化为与 data 同形的全 1 数组（通常用于输出节点）。
            - 当前实现假设单输入单输出，若 creator 涉及多输入，需扩展 Function.backward。
            - 使用列表模拟队列，按深度优先顺序处理计算图中的函数节点。
        """
        if self.grad is None:  # 如果梯度为 None，初始化为 1
            self.grad = np.ones_like(self.data)

        funcs: list[Function] = [self.creator]
        while funcs:
            f: Function = funcs.pop()  # 获取最后一个元素
            x: Variable
            y: Variable
            x, y = f.inputs, f.outputs  # 获取输入和输出
            x.grad = f.backward(y.grad)  # 计算梯度

            if x.creator is not None:
                funcs.append(x.creator)  # 将输入的创建者添加到列表中


def as_array(x: Any) -> np.ndarray:
    """
    将输入数据转换为 NumPy 数组（ndarray）类型。

    该函数检查输入是否为标量，若是标量则将其包装为 NumPy 数组；若已是数组，则直接返回。适用于统一数据类型以便后续计算。

    Args:
        x (Any): 输入数据，可以是标量（如 int、float）或数组（如 list、np.ndarray）。

    Returns:
        np.ndarray: 转换后的 NumPy 数组。若输入已是 np.ndarray，则返回原对象；若输入为标量，则返回单元素数组。

    Notes:
        - 使用 np.isscalar 判断输入是否为标量，支持 Python 内置标量类型（如 int、float）以及 NumPy 标量。
        - 若输入为非标量且不是 np.ndarray（如 list），则直接返回，可能导致类型不一致的风险。
        - 为确保健壮性，建议在调用前验证输入是否符合预期类型。

    Examples:
        >>> as_array(3)
        array([3])
        >>> as_array([1, 2, 3])
        [1, 2, 3]  # 未转换，仍为 list
        >>> as_array(np.array([1, 2]))
        array([1, 2])  # 原样返回
    """
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """
    表示一个可微函数的基类，用于构建计算图并支持前向计算与反向传播。

    该类定义了函数的基本行为，包括接受输入变量、执行前向计算并生成输出变量，以及为反向传播提供接口。通过继承此类，用户可以实现特定的函数逻辑，例如线性变换或激活函数。

    Methods:
        __call__: 执行前向计算，将输入变量转换为输出变量，并记录计算图关系。
        forward: 执行前向计算的具体逻辑，由子类实现。
        backward: 执行反向传播的具体逻辑，由子类实现。

    Notes:
        - 该类假设输入和输出均为 Variable 实例，Variable.data 为可计算的数据（如标量或张量）。
        - 子类必须实现 forward 和 backward 方法，否则调用时将抛出 NotImplementedError。
        - 当前实现假设单输出场景，若需支持多输出，需确保 forward 返回列表并在 __call__ 中正确处理。
    """

    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        """
        执行函数的前向计算，接受输入变量并返回输出变量，同时构建计算图。

        Args:
            inputs (List[Variable]): 输入变量的列表，每个元素是一个 Variable 实例，包含数据（如标量或张量）。

        Returns:
            List[Variable]: 输出变量的列表，每个元素是一个 Variable 实例，表示函数计算结果。

        Notes:
            - 输入数据的提取通过列表推导式完成，假设每个 Variable 实例具有 data 属性。
            - 输出变量通过 forward 方法计算，并包装为 Variable 实例。
            - 计算图通过为每个输出设置创建者（self）以及记录输入和输出关系来构建。
        """
        xs: List[Any] = [x.data for x in inputs]  # 提取输入变量的数据
        ys: List[Any] = self.forward(xs)          # 调用子类实现的前向计算
        outputs: List[Variable] = [Variable(as_array(y)) for y in ys]  # 将结果转换为 Variable 实例

        for output in outputs:
            output.set_creator(self)   # 为输出设置创建者，建立计算图
        self.inputs: List[Variable] = inputs  # 保存输入变量
        self.outputs: List[Variable] = outputs  # 保存输出变量
        return outputs

    def forward(self, xs: List[Any]) -> List[Any]:
        """
        执行函数的具体前向计算逻辑。

        Args:
            xs (List[Any]): 输入数据的列表，来自 Variable 实例的 data 属性（如标量或张量）。

        Returns:
            List[Any]: 前向计算的结果列表，表示函数的输出数据。

        Raises:
            NotImplementedError: 如果子类未实现该方法，则抛出此异常。

        Notes:
            - 该方法为抽象方法，需由子类具体实现，例如实现 f(x) = x^2 或其他函数逻辑。
            - 返回值应与 __call__ 中后续处理的期望一致，通常为标量或张量的列表。
        """
        raise NotImplementedError()

    def backward(self, gys: List[Any]) -> List[Any]:
        """
        执行函数的具体反向传播逻辑，计算输入的梯度。

        Args:
            gys (List[Any]): 输出梯度的列表，表示从后续节点传回的梯度值（通常与 outputs 对应）。

        Returns:
            List[Any]: 输入梯度的列表，表示对每个输入变量的偏导数。

        Raises:
            NotImplementedError: 如果子类未实现该方法，则抛出此异常。

        Notes:
            - 该方法为抽象方法，需由子类实现具体的梯度计算逻辑。
            - 输入梯度的计算应基于链式法则，利用 gys 和前向计算中的中间结果。
        """
        raise NotImplementedError()
    
class Add(Function):
    def forward(self, xs: List[Any]) -> List[Any]:
        x0, x1 = xs
        y = x0 + x1
        return(y,)
    
xs = [Variable(np.array(2.0)), Variable(np.array(3.0))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)