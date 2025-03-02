import numpy as np

class Variable:
    # 在初始化时，将传来的参数设置为实例变量data
    def __init__(self, data):
        # 严格类型检查，只接受 np.ndarray
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} is not supported")
        self.data = data