'''
本文件改进了step07.py：
1. 将反向传播计算由递归变为循环
2. 抽象def square(x)和def exp(x)，使它们更通用
3. 添加了y.grad = np.array(1.0)，使y的梯度为1
4. 只支持ndarray类型，不支持标量类型
'''
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad = None
        self.creator = None

    def sef_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:  # 如果梯度为None，初始化为1
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 获取最后一个元素
            x, y = f.input, f.output # 获取输入和输出
            x.grad = f.backward(y.grad) # 计算梯度

            if x.creator is not None:
                funcs.append(x.creator) # 将输入的创造者添加到列表中


def as_array(x):
    '''
    将输入转换为ndarray类型
    '''
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 正向传播
        output = Variable(as_array(y)) # 创建输出变量
        output.sef_creator(self) # 让输出变量保存创造者信息，即当前函数
        self.input = input # 保存输入变量
        self.output = output # 保存输出变量
        return output

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        '''
        反向传播的计算：
        1. 计算平方的导数：2x
        2. 将导数乘以输入的梯度：2x * gy
        3. 返回结果
        '''
        x = self.input.data
        gx = 2 * x * gy

        return gx
    
def square(x):
    return Square()(x)

class Exp(Function):
    def forward(self, x):

        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        
        return gx

def exp(x):
    return Exp()(x)

# # 测试
# x = Variable(np.array(1.0))
# y = square(exp(x))
# y.backward()
# print(x.grad)
