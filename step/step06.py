import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input # 保存输入的变量
        # input 设置为实例变量，而不是局部变量，是为了在 backward 方法中使用
        # 如果设置为局部变量，则无法在 backward 方法中使用


        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
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

class Exp(Function):
    def forward(self, x):

        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        
        return gx
    
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
a.grad = np.array(1.0)