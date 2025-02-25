import numpy as np
from step.step02 import Function
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    