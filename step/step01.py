class Variable:
    # 在初始化时，将传来的参数设置为实例变量data
    def __init__(self, data):
        self.data = data


# # 创建Variable实例
# import numpy as np
# data = np.array(1.0)
# # 这里的x不是数据，而是一个存放数据的实体
# x = Variable(data) 
# # 通过Variable实例的data属性，可以访问存储的数据
# print(x.data) # 1.0

# x.data = np.array(2.0)
# print(x.data) # 2.0