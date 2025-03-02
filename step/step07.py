'''
本文件用于实现反向传播的自动化
即建立这样的一个机制：无论普通的计算流程（正向传播）中是什么样的计算，反向传播都能自动进行。
这样的机制我们成为：动态计算图
'''
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def sef_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator # 获取创建者
        '''
        如果创建者存在，则进行反向传播
        如果创建者不存在，则说明已经到达了计算图的起点，Variable实例是由用户提供的变量创建的
        '''
        if f is not None: 
            x = f.input # 获取输入
            x.grad = f.backward(self.grad) # 调用backward方法，计算梯度
            x.backward() # 递归调用backward方法，计算所有输入的梯度

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 正向传播
        output = Variable(y) # 创建输出变量
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

class Exp(Function):
    def forward(self, x):

        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        
        return gx
    
square = Square()
exp = Exp()

x = Variable(np.array(0.5))
y = square(exp(x))

# 验证y的创造者是square，如果是，输出True，否则输出False
print(y.creator == square) # True

# 验证square的输入是exp
print(square.input == exp.output) # True

# 验证exp的输入是x
print(exp.input == x) # True

# 验证x的创造者是None
print(x.creator is None) # True
