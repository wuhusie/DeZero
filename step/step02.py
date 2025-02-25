from step.step01 import Variable


class Function:
    def __call__(self, input):
        '''
        input: Variable
        output: Variable

        Variable是一个包含data和grad的类
        '''
        x = input.data
        y = self.forward(x) # 具体的计算在forward方法中实现
        output = Variable(y)
        return output
    
    def forward(self, x):
        '''
        具体的计算，由子类实现。
        这里抛出异常，要求子类必须实现forward方法，否则会报错
        这是一种设计模式，叫做模板方法模式，即定义一个算法的骨架，而将一些步骤延迟到子类中实现，使得子类可以不改变一个算法的结构即可重定义该算法的某些特定步骤。
        '''
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
x = Variable(10)
f = Square()
y = f(x)
print(y)