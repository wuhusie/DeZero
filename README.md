# DeZero

DeZero 是一个简单的深度学习框架，用于学习深度学习框架的基本原理。该项目通过逐步实现来加深对深度学习框架的理解。

## 开发说明

本项目采用渐进式开发方法，通过多个步骤逐步实现深度学习框架的核心功能。每个步骤都专注于特定概念的实现，便于理解框架的工作原理。

## 第一阶段：变量类与自动求导

### 文件结构

第一阶段实现了变量类和自动求导的基本功能：

- `step01.py`: 实现了基础的 `Variable` 类，仅支持 numpy.ndarray 数据类型
- `step02.py`: 实现了 `Function` 基类和简单的 `Square` 函数
- `step03.py`: 实现了 `Exp` 函数
- `step04.py`: 实现了数值微分功能（numerical_diff）
- `step06.py`: 添加了变量的梯度属性和反向传播基础
- `step07.py`: 实现了动态计算图和自动反向传播（递归方式）
- `step09.py`: 改进了反向传播机制
  - 将递归实现改为循环实现
  - 抽象化 square 和 exp 函数接口
  - 添加梯度初始化
  - 规范化数据类型支持
- `step10.py`: 第一阶段的单元测试
  - 测试 Square 函数的正向/反向传播
  - 测试 Exp 函数的正向/反向传播
  - 实现梯度检验

### 主要特性

1. 支持基本的数学运算（平方、指数等）
2. 自动求导功能
3. 动态计算图
4. 严格的类型检查（仅支持 numpy.ndarray）
5. 完整的单元测试

### 使用示例

```python
import numpy as np
from step.step09 import Variable, square, exp

x = Variable(np.array(1.0))
y = square(exp(x))
y.backward()
print(x.grad)  # 打印梯度
```



### 运行测试

在项目根目录下运行：
```bash
python -m unittest step/step10.py
```